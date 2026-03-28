#include "lbm.cuh"

#include <algorithm>
#include <cmath>

namespace lbm {

namespace {

inline int clamp_index(int value, int lo, int hi) {
    return value < lo ? lo : (value > hi ? hi : value);
}

inline void set_host_node_type(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    int x,
    int y,
    int z,
    NodeType fill_type) {
    if (x < 0 || x >= cfg.nx || y < 0 || y >= cfg.ny || z < 0 || z >= cfg.nz) {
        return;
    }
    (*node_type)[flatten_xyz(x, y, z, cfg.nx, cfg.ny, cfg.nz)] = static_cast<std::uint8_t>(fill_type);
}

__device__ LBM_FORCEINLINE Real equilibrium(int q, Real rho, Real ux, Real uy, Real uz) {
    const Real cu = Real(g_cx[q]) * ux + Real(g_cy[q]) * uy + Real(g_cz[q]) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return g_w[q] * rho * (Real(1.0) + kInvCs2 * cu + Real(0.5) * kInvCs4 * cu * cu - Real(0.5) * kInvCs2 * uu);
}

__device__ LBM_FORCEINLINE void load_logical_cell(
    const Real* LBM_RESTRICT f,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz,
    Real* LBM_RESTRICT populations) {
    const int cell_count = nx * ny * nz;
    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const int cell = physical_cell_index(q, x, y, z, nx, ny, nz, sx, sy, sz);
        populations[q] = f[distribution_index(q, cell, cell_count)];
    }
}

__device__ LBM_FORCEINLINE void store_logical_population(
    Real* LBM_RESTRICT f,
    int q,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    const int* sx,
    const int* sy,
    const int* sz,
    Real value) {
    const int cell_count = nx * ny * nz;
    const int cell = physical_cell_index(q, x, y, z, nx, ny, nz, sx, sy, sz);
    f[distribution_index(q, cell, cell_count)] = value;
}

__device__ LBM_FORCEINLINE void recover_macro_from_populations(
    const Real* LBM_RESTRICT populations,
    Real force_x,
    Real force_y,
    Real force_z,
    Real* LBM_RESTRICT rho,
    Real* LBM_RESTRICT ux,
    Real* LBM_RESTRICT uy,
    Real* LBM_RESTRICT uz) {
    Real density = Real(0.0);
    Real mx = Real(0.0);
    Real my = Real(0.0);
    Real mz = Real(0.0);
    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        density += populations[q];
        mx += populations[q] * Real(g_cx[q]);
        my += populations[q] * Real(g_cy[q]);
        mz += populations[q] * Real(g_cz[q]);
    }

    density = density > Real(1.0e-20) ? density : Real(1.0e-20);
    *rho = density;
    *ux = (mx + Real(0.5) * force_x) / density;
    *uy = (my + Real(0.5) * force_y) / density;
    *uz = (mz + Real(0.5) * force_z) / density;
}

__global__ __launch_bounds__(kBlockSize) void apply_wall_bounceback_kernel(
    Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cell_count || node_type[tid] != kWall) {
        return;
    }

    const int x = tid % cfg.nx;
    const int yz = tid / cfg.nx;
    const int y = yz % cfg.ny;
    const int z = yz / cfg.ny;

    Real populations[kQ];
    load_logical_cell(f, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, populations);

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        // The wall layer is explicit. After the streamed field exists in logical
        // form, swapping q with opp(q) on wall nodes reflects populations back
        // toward adjacent fluid nodes using the same single DF storage.
        const int dst = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz);
        f[distribution_index(q, dst, cell_count)] = populations[g_opp[q]];
    }
}

__global__ __launch_bounds__(kBlockSize) void apply_pressure_boundaries_kernel(
    Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz) {
    const int yz_count = cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= yz_count) {
        return;
    }

    const int y = tid % cfg.ny;
    const int z = tid / cfg.ny;
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int inlet_tid = flatten_xyz(0, y, z, cfg.nx, cfg.ny, cfg.nz);
    const int outlet_tid = flatten_xyz(cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz);

    if (node_type[inlet_tid] == kInlet) {
        Real interior[kQ];
        load_logical_cell(f, 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior);

        Real rho_i = Real(0.0);
        Real ux_i = Real(0.0);
        Real uy_i = Real(0.0);
        Real uz_i = Real(0.0);
        recover_macro_from_populations(interior, Real(0.0), Real(0.0), Real(0.0), &rho_i, &ux_i, &uy_i, &uz_i);

        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            // Non-equilibrium extrapolation is applied to the whole boundary
            // node so the boundary plane is fully refreshed after the streamed
            // step and periodic wrap contamination in the incoming set is
            // removed from all boundary populations.
            const Real feq_boundary = equilibrium(q, cfg.rho_inlet, ux_i, uy_i, uz_i);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(
                f, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
        }
    }

    if (node_type[outlet_tid] == kOutlet) {
        Real interior[kQ];
        load_logical_cell(f, cfg.nx - 2, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior);

        Real rho_i = Real(0.0);
        Real ux_i = Real(0.0);
        Real uy_i = Real(0.0);
        Real uz_i = Real(0.0);
        recover_macro_from_populations(interior, Real(0.0), Real(0.0), Real(0.0), &rho_i, &ux_i, &uy_i, &uz_i);

        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            const Real feq_boundary = equilibrium(q, cfg.rho_outlet, ux_i, uy_i, uz_i);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(
                f, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
        }
    }
}

__global__ __launch_bounds__(kBlockSize) void apply_velocity_boundaries_kernel(
    Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz) {
    const int yz_count = cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= yz_count) {
        return;
    }

    const int y = tid % cfg.ny;
    const int z = tid / cfg.ny;
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int inlet_tid = flatten_xyz(0, y, z, cfg.nx, cfg.ny, cfg.nz);
    const int outlet_tid = flatten_xyz(cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz);

    if (node_type[inlet_tid] == kInlet) {
        Real interior[kQ];
        load_logical_cell(f, 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior);

        Real rho_i = Real(0.0);
        Real ux_i = Real(0.0);
        Real uy_i = Real(0.0);
        Real uz_i = Real(0.0);
        recover_macro_from_populations(interior, Real(0.0), Real(0.0), Real(0.0), &rho_i, &ux_i, &uy_i, &uz_i);

        const Real ux_b = prescribed_inlet_velocity_x(y, z, cfg);
        const Real uy_b = Real(0.0);
        const Real uz_b = Real(0.0);
        const Real rho_b = rho_i;

        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            // Rebuild the whole inlet node from the prescribed velocity and the
            // interior non-equilibrium part so the next streamed step sees a
            // self-consistent inlet state.
            const Real feq_boundary = equilibrium(q, rho_b, ux_b, uy_b, uz_b);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(
                f, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
        }
    }

    if (node_type[outlet_tid] != kOutlet) {
        return;
    }

    Real interior[kQ];
    load_logical_cell(f, cfg.nx - 2, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior);

    if (cfg.outlet_mode == OutletMode::Extrapolation) {
        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            // Extrapolation outlet: use the full adjacent interior state on the
            // boundary plane to avoid retaining stale wrapped populations.
            store_logical_population(f, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior[q]);
        }
        return;
    }

    Real rho_i = Real(0.0);
    Real ux_i = Real(0.0);
    Real uy_i = Real(0.0);
    Real uz_i = Real(0.0);
    recover_macro_from_populations(interior, Real(0.0), Real(0.0), Real(0.0), &rho_i, &ux_i, &uy_i, &uz_i);

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const Real feq_boundary = equilibrium(q, cfg.rho0, ux_i, uy_i, uz_i);
        const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
        store_logical_population(
            f, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
    }
}

}  // namespace

void launch_apply_wall_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks = (cell_count + kBlockSize - 1) / kBlockSize;
    apply_wall_bounceback_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch apply_wall_bounceback_kernel");
}

void launch_apply_pressure_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    const int yz_count = cfg.ny * cfg.nz;
    const int blocks = (yz_count + kBlockSize - 1) / kBlockSize;
    apply_pressure_boundaries_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch apply_pressure_boundaries_kernel");
}

void launch_apply_velocity_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    const int yz_count = cfg.ny * cfg.nz;
    const int blocks = (yz_count + kBlockSize - 1) / kBlockSize;
    apply_velocity_boundaries_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch apply_velocity_boundaries_kernel");
}

void build_default_node_type_map(std::vector<std::uint8_t>* node_type, const SimulationConfig& cfg) {
    if (node_type == nullptr) {
        throw std::runtime_error("build_default_node_type_map requires a valid node map.");
    }

    node_type->assign(static_cast<std::size_t>(cfg.nx * cfg.ny * cfg.nz), static_cast<std::uint8_t>(kFluid));

    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                NodeType type = kFluid;
                if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
                    type = kWall;
                } else if (cfg.mode != StreamwiseMode::PeriodicBodyForce && x == 0) {
                    type = kInlet;
                } else if (cfg.mode != StreamwiseMode::PeriodicBodyForce && x == cfg.nx - 1) {
                    type = kOutlet;
                }
                set_host_node_type(node_type, cfg, x, y, z, type);
            }
        }
    }
}

void fill_box(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    int x0,
    int x1,
    int y0,
    int y1,
    int z0,
    int z1) {
    if (node_type == nullptr || node_type->size() != static_cast<std::size_t>(cfg.nx * cfg.ny * cfg.nz)) {
        throw std::runtime_error("fill_box requires a node map sized to nx * ny * nz.");
    }

    const int xs = clamp_index(std::min(x0, x1), 0, cfg.nx - 1);
    const int xe = clamp_index(std::max(x0, x1), 0, cfg.nx - 1);
    const int ys = clamp_index(std::min(y0, y1), 0, cfg.ny - 1);
    const int ye = clamp_index(std::max(y0, y1), 0, cfg.ny - 1);
    const int zs = clamp_index(std::min(z0, z1), 0, cfg.nz - 1);
    const int ze = clamp_index(std::max(z0, z1), 0, cfg.nz - 1);

    for (int z = zs; z <= ze; ++z) {
        for (int y = ys; y <= ye; ++y) {
            for (int x = xs; x <= xe; ++x) {
                set_host_node_type(node_type, cfg, x, y, z, fill_type);
            }
        }
    }
}

void fill_ball(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    Real center_x,
    Real center_y,
    Real center_z,
    Real radius) {
    if (node_type == nullptr || node_type->size() != static_cast<std::size_t>(cfg.nx * cfg.ny * cfg.nz)) {
        throw std::runtime_error("fill_ball requires a node map sized to nx * ny * nz.");
    }
    if (radius < Real(0.0)) {
        throw std::runtime_error("fill_ball requires a non-negative radius.");
    }

    const Real radius_sq = radius * radius;
    const int xs = clamp_index(static_cast<int>(std::floor(center_x - radius)), 0, cfg.nx - 1);
    const int xe = clamp_index(static_cast<int>(std::ceil(center_x + radius)), 0, cfg.nx - 1);
    const int ys = clamp_index(static_cast<int>(std::floor(center_y - radius)), 0, cfg.ny - 1);
    const int ye = clamp_index(static_cast<int>(std::ceil(center_y + radius)), 0, cfg.ny - 1);
    const int zs = clamp_index(static_cast<int>(std::floor(center_z - radius)), 0, cfg.nz - 1);
    const int ze = clamp_index(static_cast<int>(std::ceil(center_z + radius)), 0, cfg.nz - 1);

    for (int z = zs; z <= ze; ++z) {
        for (int y = ys; y <= ye; ++y) {
            for (int x = xs; x <= xe; ++x) {
                const Real dx = Real(x) - center_x;
                const Real dy = Real(y) - center_y;
                const Real dz = Real(z) - center_z;
                if (dx * dx + dy * dy + dz * dz <= radius_sq) {
                    set_host_node_type(node_type, cfg, x, y, z, fill_type);
                }
            }
        }
    }
}

void fill_plane(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    PrimitiveAxis normal_axis,
    int coordinate,
    int half_thickness,
    int span0_begin,
    int span0_end,
    int span1_begin,
    int span1_end) {
    if (node_type == nullptr || node_type->size() != static_cast<std::size_t>(cfg.nx * cfg.ny * cfg.nz)) {
        throw std::runtime_error("fill_plane requires a node map sized to nx * ny * nz.");
    }
    if (half_thickness < 0) {
        throw std::runtime_error("fill_plane requires a non-negative half_thickness.");
    }

    if (normal_axis == PrimitiveAxis::X) {
        fill_box(node_type, cfg, fill_type, coordinate - half_thickness, coordinate + half_thickness, span0_begin, span0_end, span1_begin, span1_end);
    } else if (normal_axis == PrimitiveAxis::Y) {
        fill_box(node_type, cfg, fill_type, span0_begin, span0_end, coordinate - half_thickness, coordinate + half_thickness, span1_begin, span1_end);
    } else {
        fill_box(node_type, cfg, fill_type, span0_begin, span0_end, span1_begin, span1_end, coordinate - half_thickness, coordinate + half_thickness);
    }
}

void fill_cylinder(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    PrimitiveAxis axis,
    Real center_a,
    Real center_b,
    Real radius,
    int axis_begin,
    int axis_end) {
    if (node_type == nullptr || node_type->size() != static_cast<std::size_t>(cfg.nx * cfg.ny * cfg.nz)) {
        throw std::runtime_error("fill_cylinder requires a node map sized to nx * ny * nz.");
    }
    if (radius < Real(0.0)) {
        throw std::runtime_error("fill_cylinder requires a non-negative radius.");
    }

    const int as = std::min(axis_begin, axis_end);
    const int ae = std::max(axis_begin, axis_end);
    const Real radius_sq = radius * radius;

    if (axis == PrimitiveAxis::X) {
        const int xs = clamp_index(as, 0, cfg.nx - 1);
        const int xe = clamp_index(ae, 0, cfg.nx - 1);
        const int ys = clamp_index(static_cast<int>(std::floor(center_a - radius)), 0, cfg.ny - 1);
        const int ye = clamp_index(static_cast<int>(std::ceil(center_a + radius)), 0, cfg.ny - 1);
        const int zs = clamp_index(static_cast<int>(std::floor(center_b - radius)), 0, cfg.nz - 1);
        const int ze = clamp_index(static_cast<int>(std::ceil(center_b + radius)), 0, cfg.nz - 1);
        for (int z = zs; z <= ze; ++z) {
            for (int y = ys; y <= ye; ++y) {
                const Real da = Real(y) - center_a;
                const Real db = Real(z) - center_b;
                if (da * da + db * db > radius_sq) {
                    continue;
                }
                for (int x = xs; x <= xe; ++x) {
                    set_host_node_type(node_type, cfg, x, y, z, fill_type);
                }
            }
        }
    } else if (axis == PrimitiveAxis::Y) {
        const int ys = clamp_index(as, 0, cfg.ny - 1);
        const int ye = clamp_index(ae, 0, cfg.ny - 1);
        const int xs = clamp_index(static_cast<int>(std::floor(center_a - radius)), 0, cfg.nx - 1);
        const int xe = clamp_index(static_cast<int>(std::ceil(center_a + radius)), 0, cfg.nx - 1);
        const int zs = clamp_index(static_cast<int>(std::floor(center_b - radius)), 0, cfg.nz - 1);
        const int ze = clamp_index(static_cast<int>(std::ceil(center_b + radius)), 0, cfg.nz - 1);
        for (int z = zs; z <= ze; ++z) {
            for (int x = xs; x <= xe; ++x) {
                const Real da = Real(x) - center_a;
                const Real db = Real(z) - center_b;
                if (da * da + db * db > radius_sq) {
                    continue;
                }
                for (int y = ys; y <= ye; ++y) {
                    set_host_node_type(node_type, cfg, x, y, z, fill_type);
                }
            }
        }
    } else {
        const int zs = clamp_index(as, 0, cfg.nz - 1);
        const int ze = clamp_index(ae, 0, cfg.nz - 1);
        const int xs = clamp_index(static_cast<int>(std::floor(center_a - radius)), 0, cfg.nx - 1);
        const int xe = clamp_index(static_cast<int>(std::ceil(center_a + radius)), 0, cfg.nx - 1);
        const int ys = clamp_index(static_cast<int>(std::floor(center_b - radius)), 0, cfg.ny - 1);
        const int ye = clamp_index(static_cast<int>(std::ceil(center_b + radius)), 0, cfg.ny - 1);
        for (int y = ys; y <= ye; ++y) {
            for (int x = xs; x <= xe; ++x) {
                const Real da = Real(x) - center_a;
                const Real db = Real(y) - center_b;
                if (da * da + db * db > radius_sq) {
                    continue;
                }
                for (int z = zs; z <= ze; ++z) {
                    set_host_node_type(node_type, cfg, x, y, z, fill_type);
                }
            }
        }
    }
}

}  // namespace lbm
