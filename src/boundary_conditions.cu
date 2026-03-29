#include "lbm.cuh"

#include <algorithm>
#include <cmath>

namespace lbm {

namespace {

__host__ inline void wall_launch_config(const SimulationConfig& cfg, dim3* grid, dim3* block) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks_1d = (cell_count + kBlockSize - 1) / kBlockSize;
    *block = dim3(static_cast<unsigned int>(kBlockSize), 1u, 1u);
    *grid = dim3(static_cast<unsigned int>(blocks_1d), 1u, 1u);
}

__host__ inline void yz_launch_config(const SimulationConfig& cfg, dim3* grid, dim3* block) {
    const int yz_count = cfg.ny * cfg.nz;
    const int blocks_1d = (yz_count + kBlockSize - 1) / kBlockSize;
    *block = dim3(static_cast<unsigned int>(kBlockSize), 1u, 1u);
    *grid = dim3(static_cast<unsigned int>(blocks_1d), 1u, 1u);
}

__device__ LBM_FORCEINLINE bool wall_thread_coordinates(
    const SimulationConfig& cfg,
    int* tid,
    int* x,
    int* y,
    int* z) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    *tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (*tid >= cell_count) {
        return false;
    }
    *x = *tid % cfg.nx;
    const int yz = *tid / cfg.nx;
    *y = yz % cfg.ny;
    *z = yz / cfg.ny;
    return true;
}

__device__ LBM_FORCEINLINE bool yz_thread_coordinates(
    const SimulationConfig& cfg,
    int* y,
    int* z) {
    const int yz_count = cfg.ny * cfg.nz;
    const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (tid >= yz_count) {
        return false;
    }
    *y = tid % cfg.ny;
    *z = tid / cfg.ny;
    return true;
}

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
    StreamingView view,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    Real* LBM_RESTRICT populations) {
    const int cell = LBM_FLATTEN_XYZ(x, y, z, nx, ny, nz);
    if (view.offset != nullptr) {
        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            Real* const LBM_RESTRICT population = population_pointer(view, q);
            int physical_cell = cell + view.offset[q];
            if (physical_cell >= view.logical_cells) {
                physical_cell -= view.logical_cells;
            }
            populations[q] = population[physical_cell];
        }
        return;
    }

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        populations[q] = population_pointer(view, q)[cell];
    }
}

__device__ LBM_FORCEINLINE void store_logical_population(
    StreamingView view,
    int q,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    Real value) {
    const int cell = LBM_FLATTEN_XYZ(x, y, z, nx, ny, nz);
    if (view.offset != nullptr) {
        Real* const LBM_RESTRICT population = population_pointer(view, q);
        int physical_cell = cell + view.offset[q];
        if (physical_cell >= view.logical_cells) {
            physical_cell -= view.logical_cells;
        }
        population[physical_cell] = value;
        return;
    }

    population_pointer(view, q)[cell] = value;
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

__global__ __launch_bounds__(kLaunchBounds) void apply_wall_bounceback_kernel(
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg) {
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!wall_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }
    if (node_type[tid] != kWall) {
        return;
    }

    Real populations[kQ];
    load_logical_cell(view, x, y, z, cfg.nx, cfg.ny, cfg.nz, populations);

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        // The wall layer is explicit. After the streamed field exists in logical
        // form, swapping q with opp(q) on wall nodes reflects populations back
        // toward adjacent fluid nodes using the same single DF storage.
        store_logical_population(view, q, x, y, z, cfg.nx, cfg.ny, cfg.nz, populations[g_opp[q]]);
    }
}

__global__ __launch_bounds__(kLaunchBounds) void apply_pressure_boundaries_kernel(
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg) {
    int y = 0;
    int z = 0;
    if (!yz_thread_coordinates(cfg, &y, &z)) {
        return;
    }
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int inlet_tid = LBM_FLATTEN_XYZ(0, y, z, cfg.nx, cfg.ny, cfg.nz);
    const int outlet_tid = LBM_FLATTEN_XYZ(cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz);

    if (node_type[inlet_tid] == kInlet) {
        Real interior[kQ];
        load_logical_cell(view, 1, y, z, cfg.nx, cfg.ny, cfg.nz, interior);

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
            store_logical_population(view, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, feq_boundary + (interior[q] - feq_interior));
        }
    }

    if (node_type[outlet_tid] == kOutlet) {
        Real interior[kQ];
        load_logical_cell(view, cfg.nx - 2, y, z, cfg.nx, cfg.ny, cfg.nz, interior);

        Real rho_i = Real(0.0);
        Real ux_i = Real(0.0);
        Real uy_i = Real(0.0);
        Real uz_i = Real(0.0);
        recover_macro_from_populations(interior, Real(0.0), Real(0.0), Real(0.0), &rho_i, &ux_i, &uy_i, &uz_i);

        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            const Real feq_boundary = equilibrium(q, cfg.rho_outlet, ux_i, uy_i, uz_i);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(view, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, feq_boundary + (interior[q] - feq_interior));
        }
    }
}

__global__ __launch_bounds__(kLaunchBounds) void gather_periodic_x_planes_kernel(
    StreamingView view,
    SimulationConfig cfg,
    Real* LBM_RESTRICT xmin_plane,
    Real* LBM_RESTRICT xmax_plane) {
    int y = 0;
    int z = 0;
    if (!yz_thread_coordinates(cfg, &y, &z)) {
        return;
    }
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int yz_count = cfg.ny * cfg.nz;
    const int yz_cell = z * cfg.ny + y;
    Real xmin_pop[kQ];
    Real xmax_pop[kQ];
    Real boundary_xmin[kQ];
    Real boundary_xmax[kQ];
    load_logical_cell(view, 0, y + 1, z, cfg.nx, cfg.ny, cfg.nz, xmin_pop);
    load_logical_cell(view, cfg.nx - 1, y - 1, z, cfg.nx, cfg.ny, cfg.nz, xmax_pop);
    load_logical_cell(view, 0, y, z, cfg.nx, cfg.ny, cfg.nz, boundary_xmin);
    load_logical_cell(view, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, boundary_xmax);

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        xmin_plane[q * yz_count + yz_cell] = (g_cx[q] > 0) ? xmin_pop[q] : boundary_xmin[q];
        xmax_plane[q * yz_count + yz_cell] = (g_cx[q] < 0) ? xmax_pop[q] : boundary_xmax[q];
    }
}

__global__ __launch_bounds__(kLaunchBounds) void scatter_periodic_x_planes_kernel(
    StreamingView view,
    SimulationConfig cfg,
    const Real* LBM_RESTRICT xmin_plane,
    const Real* LBM_RESTRICT xmax_plane) {
    int y = 0;
    int z = 0;
    if (!yz_thread_coordinates(cfg, &y, &z)) {
        return;
    }
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int yz_count = cfg.ny * cfg.nz;
    const int yz_cell = z * cfg.ny + y;
    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        if (g_cx[q] > 0) {
            store_logical_population(view, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, xmin_plane[q * yz_count + yz_cell]);
        } else if (g_cx[q] < 0) {
            store_logical_population(view, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, xmax_plane[q * yz_count + yz_cell]);
        }
    }
}

__global__ __launch_bounds__(kLaunchBounds) void apply_velocity_boundaries_kernel(
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg) {
    int y = 0;
    int z = 0;
    if (!yz_thread_coordinates(cfg, &y, &z)) {
        return;
    }
    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    const int inlet_tid = LBM_FLATTEN_XYZ(0, y, z, cfg.nx, cfg.ny, cfg.nz);
    const int outlet_tid = LBM_FLATTEN_XYZ(cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz);

    if (node_type[inlet_tid] == kInlet) {
        Real interior[kQ];
        load_logical_cell(view, 1, y, z, cfg.nx, cfg.ny, cfg.nz, interior);

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
            store_logical_population(view, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, feq_boundary + (interior[q] - feq_interior));
        }
    }

    if (node_type[outlet_tid] != kOutlet) {
        return;
    }

    Real interior[kQ];
    load_logical_cell(view, cfg.nx - 2, y, z, cfg.nx, cfg.ny, cfg.nz, interior);

    if (cfg.outlet_mode == OutletMode::Extrapolation) {
        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            // Extrapolation outlet: use the full adjacent interior state on the
            // boundary plane to avoid retaining stale wrapped populations.
            store_logical_population(view, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, interior[q]);
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
        store_logical_population(view, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, feq_boundary + (interior[q] - feq_interior));
    }
}

}  // namespace

void launch_apply_wall_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg) {
    dim3 grid{};
    dim3 block{};
    wall_launch_config(cfg, &grid, &block);
    apply_wall_bounceback_kernel<<<grid, block>>>(view, d_node_type, cfg);
    cuda_check(cudaGetLastError(), "launch apply_wall_bounceback_kernel");
}

void launch_apply_pressure_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg) {
    dim3 grid{};
    dim3 block{};
    yz_launch_config(cfg, &grid, &block);
    apply_pressure_boundaries_kernel<<<grid, block>>>(view, d_node_type, cfg);
    cuda_check(cudaGetLastError(), "launch apply_pressure_boundaries_kernel");
}

void launch_apply_periodic_x_boundaries(
    StreamingView view,
    const SimulationConfig& cfg,
    Real* d_xmin_plane,
    Real* d_xmax_plane) {
    dim3 grid{};
    dim3 block{};
    yz_launch_config(cfg, &grid, &block);
    gather_periodic_x_planes_kernel<<<grid, block>>>(view, cfg, d_xmin_plane, d_xmax_plane);
    cuda_check(cudaGetLastError(), "launch gather_periodic_x_planes_kernel");
    scatter_periodic_x_planes_kernel<<<grid, block>>>(view, cfg, d_xmin_plane, d_xmax_plane);
    cuda_check(cudaGetLastError(), "launch scatter_periodic_x_planes_kernel");
}

void launch_apply_velocity_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg) {
    dim3 grid{};
    dim3 block{};
    yz_launch_config(cfg, &grid, &block);
    apply_velocity_boundaries_kernel<<<grid, block>>>(view, d_node_type, cfg);
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
