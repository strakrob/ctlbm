#include "lbm.cuh"

namespace lbm {

__device__ __constant__ int g_cx[kQ];
__device__ __constant__ int g_cy[kQ];
__device__ __constant__ int g_cz[kQ];
__device__ __constant__ int g_opp[kQ];
__device__ __constant__ Real g_w[kQ];

namespace {

__device__ inline Real equilibrium(int q, Real rho, Real ux, Real uy, Real uz) {
    const Real cu = Real(g_cx[q]) * ux + Real(g_cy[q]) * uy + Real(g_cz[q]) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return g_w[q] * rho * (Real(1.0) + kInvCs2 * cu + Real(0.5) * kInvCs4 * cu * cu - Real(0.5) * kInvCs2 * uu);
}

__device__ inline Real guo_force_term(int q, Real ux, Real uy, Real uz, Real fx, Real fy, Real fz, Real omega) {
    const Real cu = Real(g_cx[q]) * ux + Real(g_cy[q]) * uy + Real(g_cz[q]) * uz;
    const Real c_minus_u_x = Real(g_cx[q]) - ux;
    const Real c_minus_u_y = Real(g_cy[q]) - uy;
    const Real c_minus_u_z = Real(g_cz[q]) - uz;

    const Real projection_x = c_minus_u_x * kInvCs2 + Real(g_cx[q]) * cu * kInvCs4;
    const Real projection_y = c_minus_u_y * kInvCs2 + Real(g_cy[q]) * cu * kInvCs4;
    const Real projection_z = c_minus_u_z * kInvCs2 + Real(g_cz[q]) * cu * kInvCs4;
    const Real forcing_projection = projection_x * fx + projection_y * fy + projection_z * fz;
    return g_w[q] * (Real(1.0) - Real(0.5) * omega) * forcing_projection;
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
        // The logical field for direction q is stored with its own running shift.
        const int cell = physical_cell_index(q, x, y, z, nx, ny, nz, sx, sy, sz);
        populations[q] = f[distribution_index(q, cell, cell_count)];
    }
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
        const Real fq = populations[q];
        density += fq;
        mx += fq * Real(g_cx[q]);
        my += fq * Real(g_cy[q]);
        mz += fq * Real(g_cz[q]);
    }

    density = density > Real(1.0e-20) ? density : Real(1.0e-20);
    *rho = density;
    *ux = (mx + Real(0.5) * force_x) / density;
    *uy = (my + Real(0.5) * force_y) / density;
    *uz = (mz + Real(0.5) * force_z) / density;
}

__global__ void classify_nodes_kernel(std::uint8_t* node_type, SimulationConfig cfg) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cell_count) {
        return;
    }

    const int x = tid % cfg.nx;
    const int yz = tid / cfg.nx;
    const int y = yz % cfg.ny;
    const int z = yz / cfg.ny;

    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        node_type[tid] = kWall;
        return;
    }

    if (cfg.mode == StreamwiseMode::PeriodicBodyForce) {
        node_type[tid] = kFluid;
        return;
    }

    if (x == 0) {
        node_type[tid] = kInlet;
    } else if (x == cfg.nx - 1) {
        node_type[tid] = kOutlet;
    } else {
        node_type[tid] = kFluid;
    }
}

__global__ void initialize_equilibrium_kernel(
    Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* sx,
    const int* sy,
    const int* sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cell_count) {
        return;
    }

    const int x = tid % cfg.nx;
    const int yz = tid / cfg.nx;
    const int y = yz % cfg.ny;
    const int z = yz / cfg.ny;

    // Boundary planes start from their target state so the first few steps do not
    // begin from a strongly inconsistent inlet or outlet field.
    Real rho = cfg.rho0;
    Real ux = Real(0.0);
    Real uy = Real(0.0);
    Real uz = Real(0.0);

    if (node_type[tid] == kInlet && cfg.mode == StreamwiseMode::Pressure) {
        rho = cfg.rho_inlet;
    } else if (node_type[tid] == kOutlet && cfg.mode == StreamwiseMode::Pressure) {
        rho = cfg.rho_outlet;
    } else if (node_type[tid] == kInlet && cfg.mode == StreamwiseMode::Velocity) {
        ux = prescribed_inlet_velocity_x(y, z, cfg);
    }

    for (int q = 0; q < kQ; ++q) {
        const int cell = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz);
        f[distribution_index(q, cell, cell_count)] = equilibrium(q, rho, ux, uy, uz);
    }
}

__global__ void collide_and_stream_kernel(
    Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* current_sx,
    const int* current_sy,
    const int* current_sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cell_count) {
        return;
    }

    const int x = tid % cfg.nx;
    const int yz = tid / cfg.nx;
    const int y = yz % cfg.ny;
    const int z = yz / cfg.ny;

    Real populations[kQ];
    load_logical_cell(f, x, y, z, cfg.nx, cfg.ny, cfg.nz, current_sx, current_sy, current_sz, populations);

    if (node_type[tid] != kFluid) {
        // Walls and explicit inlet/outlet planes are not collided as ordinary
        // fluid nodes. They are carried forward into the new logical shift
        // state and then overwritten by dedicated boundary kernels.
        for (int q = 0; q < kQ; ++q) {
            const int dst = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, current_sx, current_sy, current_sz);
            f[distribution_index(q, dst, cell_count)] = populations[q];
        }
        return;
    }

    const Real fx = (cfg.mode == StreamwiseMode::PeriodicBodyForce) ? cfg.body_force_x : Real(0.0);
    const Real fy = Real(0.0);
    const Real fz = Real(0.0);

    Real rho = Real(0.0);
    Real ux = Real(0.0);
    Real uy = Real(0.0);
    Real uz = Real(0.0);
    recover_macro_from_populations(populations, fx, fy, fz, &rho, &ux, &uy, &uz);

    for (int q = 0; q < kQ; ++q) {
        const Real feq = equilibrium(q, rho, ux, uy, uz);
        const Real force_term = guo_force_term(q, ux, uy, uz, fx, fy, fz, cfg.omega);
        const Real post_collision = populations[q] - cfg.omega * (populations[q] - feq) + force_term;
        // In periodic-shift streaming the post-collision value stays in the
        // current physical slot. The host advances the logical shifts after the
        // kernel, which makes the next logical access observe the streamed
        // neighbour without a second DF array.
        const int dst = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, current_sx, current_sy, current_sz);
        f[distribution_index(q, dst, cell_count)] = post_collision;
    }
}

__global__ void recover_macros_kernel(
    const Real* f,
    const std::uint8_t* node_type,
    SimulationConfig cfg,
    const int* sx,
    const int* sy,
    const int* sz,
    Real* rho,
    Real* ux,
    Real* uy,
    Real* uz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= cell_count) {
        return;
    }

    const int x = tid % cfg.nx;
    const int yz = tid / cfg.nx;
    const int y = yz % cfg.ny;
    const int z = yz / cfg.ny;

    Real populations[kQ];
    load_logical_cell(f, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, populations);

    const Real fx = (cfg.mode == StreamwiseMode::PeriodicBodyForce && node_type[tid] != kWall) ? cfg.body_force_x : Real(0.0);

    Real density = Real(0.0);
    Real vx = Real(0.0);
    Real vy = Real(0.0);
    Real vz = Real(0.0);
    recover_macro_from_populations(populations, fx, Real(0.0), Real(0.0), &density, &vx, &vy, &vz);

    rho[tid] = density;
    ux[tid] = (node_type[tid] == kWall) ? Real(0.0) : vx;
    uy[tid] = (node_type[tid] == kWall) ? Real(0.0) : vy;
    uz[tid] = (node_type[tid] == kWall) ? Real(0.0) : vz;
}

}  // namespace

void copy_lattice_constants_to_device() {
    cuda_check(cudaMemcpyToSymbol(g_cx, kCx.data(), sizeof(int) * kQ), "copy D3Q27 cx constants");
    cuda_check(cudaMemcpyToSymbol(g_cy, kCy.data(), sizeof(int) * kQ), "copy D3Q27 cy constants");
    cuda_check(cudaMemcpyToSymbol(g_cz, kCz.data(), sizeof(int) * kQ), "copy D3Q27 cz constants");
    cuda_check(cudaMemcpyToSymbol(g_opp, kOpp.data(), sizeof(int) * kQ), "copy D3Q27 opposite constants");
    cuda_check(cudaMemcpyToSymbol(g_w, kWeights.data(), sizeof(Real) * kQ), "copy D3Q27 weight constants");
}

void launch_classify_nodes(std::uint8_t* d_node_type, const SimulationConfig& cfg) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks = (cell_count + kBlockSize - 1) / kBlockSize;
    classify_nodes_kernel<<<blocks, kBlockSize>>>(d_node_type, cfg);
    cuda_check(cudaGetLastError(), "launch classify_nodes_kernel");
}

void launch_initialize_equilibrium(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks = (cell_count + kBlockSize - 1) / kBlockSize;
    initialize_equilibrium_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch initialize_equilibrium_kernel");
}

void launch_collide_and_stream(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks = (cell_count + kBlockSize - 1) / kBlockSize;
    collide_and_stream_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch collide_and_stream_kernel");
}

void launch_recover_macros(
    const Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz,
    Real* d_rho,
    Real* d_ux,
    Real* d_uy,
    Real* d_uz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks = (cell_count + kBlockSize - 1) / kBlockSize;
    recover_macros_kernel<<<blocks, kBlockSize>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz, d_rho, d_ux, d_uy, d_uz);
    cuda_check(cudaGetLastError(), "launch recover_macros_kernel");
}

}  // namespace lbm
