#include "lbm.cuh"

namespace lbm {

namespace {

__device__ inline Real equilibrium(int q, Real rho, Real ux, Real uy, Real uz) {
    const Real cu = Real(g_cx[q]) * ux + Real(g_cy[q]) * uy + Real(g_cz[q]) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return g_w[q] * rho * (Real(1.0) + kInvCs2 * cu + Real(0.5) * kInvCs4 * cu * cu - Real(0.5) * kInvCs2 * uu);
}

__device__ inline void load_logical_cell(
    const Real* f,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    const int* sx,
    const int* sy,
    const int* sz,
    Real* populations) {
    const int cell_count = nx * ny * nz;
    for (int q = 0; q < kQ; ++q) {
        const int cell = physical_cell_index(q, x, y, z, nx, ny, nz, sx, sy, sz);
        populations[q] = f[distribution_index(q, cell, cell_count)];
    }
}

__device__ inline void store_logical_population(
    Real* f,
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

__device__ inline void recover_macro_from_populations(
    const Real* populations,
    Real force_x,
    Real force_y,
    Real force_z,
    Real* rho,
    Real* ux,
    Real* uy,
    Real* uz) {
    Real density = Real(0.0);
    Real mx = Real(0.0);
    Real my = Real(0.0);
    Real mz = Real(0.0);
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

__global__ void apply_wall_bounceback_kernel(
    Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* sx,
    const int* sy,
    const int* sz) {
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

    for (int q = 0; q < kQ; ++q) {
        // The wall layer is explicit. After the streamed field exists in logical
        // form, swapping q with opp(q) on wall nodes reflects populations back
        // toward adjacent fluid nodes using the same single DF storage.
        const int dst = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz);
        f[distribution_index(q, dst, cell_count)] = populations[g_opp[q]];
    }
}

__global__ void apply_pressure_boundaries_kernel(
    Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* sx,
    const int* sy,
    const int* sz) {
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

        for (int q = 0; q < kQ; ++q) {
            // Non-equilibrium extrapolation is applied to the whole boundary
            // node so the boundary plane is fully refreshed after the streamed
            // step and periodic wrap contamination in the incoming set is
            // removed from all boundary populations.
            const Real feq_boundary = equilibrium(q, cfg.rho_inlet, ux_i, uy_i, uz_i);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(f, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
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

        for (int q = 0; q < kQ; ++q) {
            const Real feq_boundary = equilibrium(q, cfg.rho_outlet, ux_i, uy_i, uz_i);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(
                f, q, cfg.nx - 1, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
        }
    }
}

__global__ void apply_velocity_boundaries_kernel(
    Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* sx,
    const int* sy,
    const int* sz) {
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

        for (int q = 0; q < kQ; ++q) {
            // Rebuild the whole inlet node from the prescribed velocity and the
            // interior non-equilibrium part so the next streamed step sees a
            // self-consistent inlet state.
            const Real feq_boundary = equilibrium(q, rho_b, ux_b, uy_b, uz_b);
            const Real feq_interior = equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            store_logical_population(f, q, 0, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, feq_boundary + (interior[q] - feq_interior));
        }
    }

    if (node_type[outlet_tid] != kOutlet) {
        return;
    }

    Real interior[kQ];
    load_logical_cell(f, cfg.nx - 2, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, interior);

    if (cfg.outlet_mode == OutletMode::Extrapolation) {
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

}  // namespace lbm
