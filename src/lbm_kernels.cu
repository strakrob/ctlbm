#include "lbm.cuh"

namespace lbm {

__device__ __constant__ int g_cx[kQ];
__device__ __constant__ int g_cy[kQ];
__device__ __constant__ int g_cz[kQ];
__device__ __constant__ int g_opp[kQ];
__device__ __constant__ Real g_w[kQ];

namespace {

__host__ inline void volume_launch_config(const SimulationConfig& cfg, dim3* grid, dim3* block) {
#if defined(LBM_USE_3D_TOPOLOGY)
    if (cfg.nx <= 0 || cfg.nx > kCudaMaxThreadsPerBlock) {
        throw std::runtime_error("LBM_USE_3D_TOPOLOGY requires nx in [1, 1024].");
    }
    *block = dim3(static_cast<unsigned int>(cfg.nx), 1u, 1u);
    *grid = dim3(static_cast<unsigned int>(cfg.ny), static_cast<unsigned int>(cfg.nz), 1u);
#else
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks_1d = (cell_count + kBlockSize - 1) / kBlockSize;
    *block = dim3(static_cast<unsigned int>(kBlockSize), 1u, 1u);
    *grid = dim3(static_cast<unsigned int>(blocks_1d), 1u, 1u);
#endif
}

__device__ LBM_FORCEINLINE bool volume_thread_coordinates(
    const SimulationConfig& cfg,
    int* tid,
    int* x,
    int* y,
    int* z) {
#if defined(LBM_USE_3D_TOPOLOGY)
    *x = static_cast<int>(threadIdx.x);
    *y = static_cast<int>(blockIdx.x);
    *z = static_cast<int>(blockIdx.y);
    if (*x >= cfg.nx || *y >= cfg.ny || *z >= cfg.nz) {
        return false;
    }
    *tid = LBM_FLATTEN_XYZ(*x, *y, *z, cfg.nx, cfg.ny, cfg.nz);
    return true;
#else
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
#endif
}

__device__ LBM_FORCEINLINE void load_shift_vectors_to_shared(
    const int* LBM_RESTRICT src_x,
    const int* LBM_RESTRICT src_y,
    const int* LBM_RESTRICT src_z,
    int* LBM_RESTRICT dst_x,
    int* LBM_RESTRICT dst_y,
    int* LBM_RESTRICT dst_z) {
    for (int q = static_cast<int>(threadIdx.x); q < kQ; q += static_cast<int>(blockDim.x)) {
        dst_x[q] = src_x[q];
        dst_y[q] = src_y[q];
        dst_z[q] = src_z[q];
    }
}

__device__ LBM_FORCEINLINE Real equilibrium(int q, Real rho, Real ux, Real uy, Real uz) {
    const Real cu = Real(g_cx[q]) * ux + Real(g_cy[q]) * uy + Real(g_cz[q]) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return g_w[q] * rho * (Real(1.0) + kInvCs2 * cu + Real(0.5) * kInvCs4 * cu * cu - Real(0.5) * kInvCs2 * uu);
}

__device__ LBM_FORCEINLINE Real guo_force_term(int q, Real ux, Real uy, Real uz, Real fx, Real fy, Real fz, Real omega) {
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
        // The logical field for direction q is stored with its own running shift.
        const int cell = physical_cell_index(q, x, y, z, nx, ny, nz, sx, sy, sz);
        populations[q] = f[LBM_DISTRIBUTION_INDEX(q, cell, cell_count)];
    }
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

__global__ __launch_bounds__(kLaunchBounds) void classify_nodes_kernel(std::uint8_t* LBM_RESTRICT node_type, SimulationConfig cfg) {
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!volume_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }

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

__global__ __launch_bounds__(kLaunchBounds) void initialize_equilibrium_kernel(
    Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!volume_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }

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

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const int cell = physical_cell_index(q, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz);
        f[LBM_DISTRIBUTION_INDEX(q, cell, cell_count)] = equilibrium(q, rho, ux, uy, uz);
    }
}

__global__ __launch_bounds__(kLaunchBounds) void collide_and_stream_kernel(
    Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT current_sx,
    const int* LBM_RESTRICT current_sy,
    const int* LBM_RESTRICT current_sz) {
    __shared__ int shared_sx[kQ];
    __shared__ int shared_sy[kQ];
    __shared__ int shared_sz[kQ];
    load_shift_vectors_to_shared(current_sx, current_sy, current_sz, shared_sx, shared_sy, shared_sz);
    __syncthreads();

    const int nx = cfg.nx;
    const int ny = cfg.ny;
    const int nz = cfg.nz;
    const int cell_count = nx * ny * nz;
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!volume_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }

    Real populations[kQ];
    load_logical_cell(f, x, y, z, nx, ny, nz, shared_sx, shared_sy, shared_sz, populations);

    const std::uint8_t type = node_type[tid];

    if (type != kFluid) {
        // Walls and explicit inlet/outlet planes are not collided as ordinary
        // fluid nodes. They are carried forward into the new logical shift
        // state and then overwritten by dedicated boundary kernels.
        #pragma unroll
        for (int q = 0; q < kQ; ++q) {
            const int dst = physical_cell_index(q, x, y, z, nx, ny, nz, shared_sx, shared_sy, shared_sz);
            f[LBM_DISTRIBUTION_INDEX(q, dst, cell_count)] = populations[q];
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

    #pragma unroll
    for (int q = 0; q < kQ; ++q) {
        const Real feq = equilibrium(q, rho, ux, uy, uz);
        const Real force_term = guo_force_term(q, ux, uy, uz, fx, fy, fz, cfg.omega);
        const Real post_collision = populations[q] - cfg.omega * (populations[q] - feq) + force_term;
        // In periodic-shift streaming the post-collision value stays in the
        // current physical slot. The host advances the logical shifts after the
        // kernel, which makes the next logical access observe the streamed
        // neighbour without a second DF array.
        const int dst = physical_cell_index(q, x, y, z, nx, ny, nz, shared_sx, shared_sy, shared_sz);
        f[LBM_DISTRIBUTION_INDEX(q, dst, cell_count)] = post_collision;
    }
}

__global__ __launch_bounds__(kLaunchBounds) void recover_macros_kernel(
    const Real* LBM_RESTRICT f,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
    const int* LBM_RESTRICT sx,
    const int* LBM_RESTRICT sy,
    const int* LBM_RESTRICT sz,
    Real* LBM_RESTRICT rho,
    Real* LBM_RESTRICT ux,
    Real* LBM_RESTRICT uy,
    Real* LBM_RESTRICT uz) {
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!volume_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }

    Real populations[kQ];
    load_logical_cell(f, x, y, z, cfg.nx, cfg.ny, cfg.nz, sx, sy, sz, populations);

    const std::uint8_t type = node_type[tid];
    const Real fx = (cfg.mode == StreamwiseMode::PeriodicBodyForce && type != kWall) ? cfg.body_force_x : Real(0.0);

    Real density = Real(0.0);
    Real vx = Real(0.0);
    Real vy = Real(0.0);
    Real vz = Real(0.0);
    recover_macro_from_populations(populations, fx, Real(0.0), Real(0.0), &density, &vx, &vy, &vz);

    rho[tid] = density;
    ux[tid] = (type == kWall) ? Real(0.0) : vx;
    uy[tid] = (type == kWall) ? Real(0.0) : vy;
    uz[tid] = (type == kWall) ? Real(0.0) : vz;
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
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    classify_nodes_kernel<<<grid, block>>>(d_node_type, cfg);
    cuda_check(cudaGetLastError(), "launch classify_nodes_kernel");
}

void launch_initialize_equilibrium(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    initialize_equilibrium_kernel<<<grid, block>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
    cuda_check(cudaGetLastError(), "launch initialize_equilibrium_kernel");
}

void launch_collide_and_stream(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz) {
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    collide_and_stream_kernel<<<grid, block>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
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
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    recover_macros_kernel<<<grid, block>>>(d_f, d_node_type, cfg, d_sx, d_sy, d_sz, d_rho, d_ux, d_uy, d_uz);
    cuda_check(cudaGetLastError(), "launch recover_macros_kernel");
}

}  // namespace lbm
