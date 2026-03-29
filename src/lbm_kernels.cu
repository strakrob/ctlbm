#include "lbm.cuh"

namespace lbm {

#define iCsq (3.0)
#define Wc (8./27.)
#define Ws (2./27.)
#define Wm (1./54.)
#define Wl (1./216.)
#define _dfeq(q1,q2,q3) (rho * (1. -.5*iCsq * uu + iCsq*(q1*ux + q2*uy + q3*uz)*( 1. + .5*iCsq*(q1*ux + q2*uy + q3*uz))))

// LBM_FORCEINLINE for feq worsens MLUPS !?
__device__  Real _feq_zzz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wc*_dfeq( 0, 0, 0);}
__device__  Real _feq_pzz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq( 1, 0, 0);}
__device__  Real _feq_mzz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq(-1, 0, 0);}
__device__  Real _feq_zpz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq( 0, 1, 0);}
__device__  Real _feq_zmz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq( 0,-1, 0);}
__device__  Real _feq_zzp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq( 0, 0, 1);}
__device__  Real _feq_zzm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Ws*_dfeq( 0, 0,-1);}
__device__  Real _feq_ppz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 1, 1, 0);}
__device__  Real _feq_pmz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 1,-1, 0);}
__device__  Real _feq_mpz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq(-1, 1, 0);}
__device__  Real _feq_mmz(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq(-1,-1, 0);}
__device__  Real _feq_pzp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 1, 0, 1);}
__device__  Real _feq_mzm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq(-1, 0,-1);}
__device__  Real _feq_pzm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 1, 0,-1);}
__device__  Real _feq_mzp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq(-1, 0, 1);}
__device__  Real _feq_zpp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 0, 1, 1);}
__device__  Real _feq_zpm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 0, 1,-1);}
__device__  Real _feq_zmp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 0,-1, 1);}
__device__  Real _feq_zmm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wm*_dfeq( 0,-1,-1);}
__device__  Real _feq_ppp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq( 1, 1, 1);}
__device__  Real _feq_mmm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq(-1,-1,-1);}
__device__  Real _feq_ppm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq( 1, 1,-1);}
__device__  Real _feq_pmp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq( 1,-1, 1);}
__device__  Real _feq_mpp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq(-1, 1, 1);}
__device__  Real _feq_mpm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq(-1, 1,-1);}
__device__  Real _feq_mmp(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq(-1,-1, 1);}
__device__  Real _feq_pmm(Real rho, Real ux, Real uy, Real uz, Real uu) {return Wl*_dfeq( 1,-1,-1);}

__device__ __constant__ int g_cx[kQ];
__device__ __constant__ int g_cy[kQ];
__device__ __constant__ int g_cz[kQ];
__device__ __constant__ int g_opp[kQ];
__device__ __constant__ Real g_w[kQ];

#define LBM_VMM_OFFSET_CELL(cell, offset, logical_cells) (((cell) + (offset) < (logical_cells)) ? ((cell) + (offset)) : ((cell) + (offset) - (logical_cells)))
#define LBM_LOAD_F_ALIGNED(q) Real f_##q = view.p##q[(cell)]
#define LBM_LOAD_F_OFFSET(q) Real f_##q = view.p##q[LBM_VMM_OFFSET_CELL((cell), offset[(q)], logical_cells)]
#define LBM_STORE_F_ALIGNED(q) view.p##q[(cell)] = f_##q
#define LBM_STORE_F_OFFSET(q) view.p##q[LBM_VMM_OFFSET_CELL((cell), offset[(q)], logical_cells)] = f_##q
#if defined(LBM_BENCH_COMPUTE_ONLY)
#define LBM_BENCH_ACCUM_DECL Real bench_checksum = 0.0
#define LBM_BENCH_ACCUMULATE(value) bench_checksum += (value) * 1.0e-30
#define LBM_BENCH_FLUSH_ALIGNED() do { if (bench_checksum == -1234567.0) view.p0[(cell)] = bench_checksum; } while (0)
#define LBM_BENCH_FLUSH_OFFSET() do { if (bench_checksum == -1234567.0) view.p0[LBM_VMM_OFFSET_CELL((cell), offset[0], logical_cells)] = bench_checksum; } while (0)
#else
#define LBM_BENCH_ACCUM_DECL
#define LBM_BENCH_ACCUMULATE(value) do { } while (0)
#define LBM_BENCH_FLUSH_ALIGNED() do { } while (0)
#define LBM_BENCH_FLUSH_OFFSET() do { } while (0)
#endif

namespace {

__host__ inline void volume_launch_config(const SimulationConfig& cfg, dim3* grid, dim3* block) {
    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    const int blocks_1d = (cell_count + kBlockSize - 1) / kBlockSize;
    *block = dim3(static_cast<unsigned int>(kBlockSize), 1u, 1u);
    *grid = dim3(static_cast<unsigned int>(blocks_1d), 1u, 1u);
}

__device__ LBM_FORCEINLINE bool volume_thread_coordinates(
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
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg) {
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
        store_logical_population(view, q, x, y, z, cfg.nx, cfg.ny, cfg.nz, equilibrium(q, rho, ux, uy, uz));
    }
}


// FIXME add force for model A
template <bool UseOffset>
__global__ __launch_bounds__(kLaunchBounds) void collide_and_stream_kernel(
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg) {
    int tid = 0;
    int x = 0;
    int y = 0;
    int z = 0;
    if (!volume_thread_coordinates(cfg, &tid, &x, &y, &z)) {
        return;
    }
    const int cell = tid;

    if constexpr (UseOffset) {
        const int* LBM_RESTRICT offset = view.offset;
        const int logical_cells = view.logical_cells;
        LBM_LOAD_F_OFFSET(0);
        LBM_LOAD_F_OFFSET(1);
        LBM_LOAD_F_OFFSET(2);
        LBM_LOAD_F_OFFSET(3);
        LBM_LOAD_F_OFFSET(4);
        LBM_LOAD_F_OFFSET(5);
        LBM_LOAD_F_OFFSET(6);
        LBM_LOAD_F_OFFSET(7);
        LBM_LOAD_F_OFFSET(8);
        LBM_LOAD_F_OFFSET(9);
        LBM_LOAD_F_OFFSET(10);
        LBM_LOAD_F_OFFSET(11);
        LBM_LOAD_F_OFFSET(12);
        LBM_LOAD_F_OFFSET(13);
        LBM_LOAD_F_OFFSET(14);
        LBM_LOAD_F_OFFSET(15);
        LBM_LOAD_F_OFFSET(16);
        LBM_LOAD_F_OFFSET(17);
        LBM_LOAD_F_OFFSET(18);
        LBM_LOAD_F_OFFSET(19);
        LBM_LOAD_F_OFFSET(20);
        LBM_LOAD_F_OFFSET(21);
        LBM_LOAD_F_OFFSET(22);
        LBM_LOAD_F_OFFSET(23);
        LBM_LOAD_F_OFFSET(24);
        LBM_LOAD_F_OFFSET(25);
        LBM_LOAD_F_OFFSET(26);
        
        const std::uint8_t type = node_type[tid];
        if (type != kFluid) {
            LBM_STORE_F_OFFSET(0);
            LBM_STORE_F_OFFSET(1);
            LBM_STORE_F_OFFSET(2);
            LBM_STORE_F_OFFSET(3);
            LBM_STORE_F_OFFSET(4);
            LBM_STORE_F_OFFSET(5);
            LBM_STORE_F_OFFSET(6);
            LBM_STORE_F_OFFSET(7);
            LBM_STORE_F_OFFSET(8);
            LBM_STORE_F_OFFSET(9);
            LBM_STORE_F_OFFSET(10);
            LBM_STORE_F_OFFSET(11);
            LBM_STORE_F_OFFSET(12);
            LBM_STORE_F_OFFSET(13);
            LBM_STORE_F_OFFSET(14);
            LBM_STORE_F_OFFSET(15);
            LBM_STORE_F_OFFSET(16);
            LBM_STORE_F_OFFSET(17);
            LBM_STORE_F_OFFSET(18);
            LBM_STORE_F_OFFSET(19);
            LBM_STORE_F_OFFSET(20);
            LBM_STORE_F_OFFSET(21);
            LBM_STORE_F_OFFSET(22);
            LBM_STORE_F_OFFSET(23);
            LBM_STORE_F_OFFSET(24);
            LBM_STORE_F_OFFSET(25);
            LBM_STORE_F_OFFSET(26);
            return;
        }

#if defined(LBM_BENCH_STORE_INPUT)
        // Memory-path benchmark: preserve the 27 DF writes but skip collision math.
#else
        const Real fx = (cfg.mode == StreamwiseMode::PeriodicBodyForce) ? cfg.body_force_x : 0.0;
        const Real fy = 0.0;
        const Real fz = 0.0;
        Real rho =
            f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 +
            f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 +
            f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
        rho = rho > 1.0e-20 ? rho : 1.0e-20;
        const Real invrho = 1.0 / rho;
        Real ux =
            f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_11 - f_12 + f_13 - f_14 +
            f_19 - f_20 + f_21 - f_22 + f_23 - f_24 + f_25 - f_26;
        Real uy =
            f_3 - f_4 + f_7 - f_8 - f_9 + f_10 + f_15 - f_16 + f_17 - f_18 +
            f_19 - f_20 + f_21 - f_22 - f_23 + f_24 - f_25 + f_26;
        Real uz =
            f_5 - f_6 + f_11 - f_12 - f_13 + f_14 + f_15 - f_16 - f_17 + f_18 +
            f_19 - f_20 - f_21 + f_22 + f_23 - f_24 - f_25 + f_26;
        ux = (ux + 0.5 * fx) * invrho;
        uy = (uy + 0.5 * fy) * invrho;
        uz = (uz + 0.5 * fz) * invrho;
        const Real uu = ux * ux + uy * uy + uz * uz;
        LBM_BENCH_ACCUM_DECL;
        const Real omega = cfg.omega;
        
		f_20 += omega*(_feq_mmm(rho,ux,uy,uz,uu) - f_20);
		f_8 += omega*(_feq_mmz(rho,ux,uy,uz,uu) - f_8);
		f_22 += omega*(_feq_mmp(rho,ux,uy,uz,uu) - f_22);
		f_12 += omega*(_feq_mzm(rho,ux,uy,uz,uu) - f_12);
		f_2 += omega*(_feq_mzz(rho,ux,uy,uz,uu) - f_2);
		f_14 += omega*(_feq_mzp(rho,ux,uy,uz,uu) - f_14);
		f_24 += omega*(_feq_mpm(rho,ux,uy,uz,uu) - f_24);
		f_10 += omega*(_feq_mpz(rho,ux,uy,uz,uu) - f_10);
		f_26 += omega*(_feq_mpp(rho,ux,uy,uz,uu) - f_26);
		f_16 += omega*(_feq_zmm(rho,ux,uy,uz,uu) - f_16);
		f_4 += omega*(_feq_zmz(rho,ux,uy,uz,uu) - f_4);
		f_18 += omega*(_feq_zmp(rho,ux,uy,uz,uu) - f_18);
		f_6 += omega*(_feq_zzm(rho,ux,uy,uz,uu) - f_6);
		f_0 += omega*(_feq_zzz(rho,ux,uy,uz,uu) - f_0);
		f_5 += omega*(_feq_zzp(rho,ux,uy,uz,uu) - f_5);
		f_17 += omega*(_feq_zpm(rho,ux,uy,uz,uu) - f_17);
		f_3 += omega*(_feq_zpz(rho,ux,uy,uz,uu) - f_3);
		f_15 += omega*(_feq_zpp(rho,ux,uy,uz,uu) - f_15);
		f_25 += omega*(_feq_pmm(rho,ux,uy,uz,uu) - f_25);
		f_9 += omega*(_feq_pmz(rho,ux,uy,uz,uu) - f_9);
		f_23 += omega*(_feq_pmp(rho,ux,uy,uz,uu) - f_23);
		f_13 += omega*(_feq_pzm(rho,ux,uy,uz,uu) - f_13);
		f_1 += omega*(_feq_pzz(rho,ux,uy,uz,uu) - f_1);
		f_11 += omega*(_feq_pzp(rho,ux,uy,uz,uu) - f_11);
		f_21 += omega*(_feq_ppm(rho,ux,uy,uz,uu) - f_21);
		f_7 += omega*(_feq_ppz(rho,ux,uy,uz,uu) - f_7);
		f_19 += omega*(_feq_ppp(rho,ux,uy,uz,uu) - f_19);
#endif
#if !defined(LBM_BENCH_COMPUTE_ONLY)
        LBM_STORE_F_OFFSET(0);
        LBM_STORE_F_OFFSET(1);
        LBM_STORE_F_OFFSET(2);
        LBM_STORE_F_OFFSET(3);
        LBM_STORE_F_OFFSET(4);
        LBM_STORE_F_OFFSET(5);
        LBM_STORE_F_OFFSET(6);
        LBM_STORE_F_OFFSET(7);
        LBM_STORE_F_OFFSET(8);
        LBM_STORE_F_OFFSET(9);
        LBM_STORE_F_OFFSET(10);
        LBM_STORE_F_OFFSET(11);
        LBM_STORE_F_OFFSET(12);
        LBM_STORE_F_OFFSET(13);
        LBM_STORE_F_OFFSET(14);
        LBM_STORE_F_OFFSET(15);
        LBM_STORE_F_OFFSET(16);
        LBM_STORE_F_OFFSET(17);
        LBM_STORE_F_OFFSET(18);
        LBM_STORE_F_OFFSET(19);
        LBM_STORE_F_OFFSET(20);
        LBM_STORE_F_OFFSET(21);
        LBM_STORE_F_OFFSET(22);
        LBM_STORE_F_OFFSET(23);
        LBM_STORE_F_OFFSET(24);
        LBM_STORE_F_OFFSET(25);
        LBM_STORE_F_OFFSET(26);
#else
        LBM_BENCH_FLUSH_OFFSET();
#endif
    } else {
        LBM_LOAD_F_ALIGNED(0);
        LBM_LOAD_F_ALIGNED(1);
        LBM_LOAD_F_ALIGNED(2);
        LBM_LOAD_F_ALIGNED(3);
        LBM_LOAD_F_ALIGNED(4);
        LBM_LOAD_F_ALIGNED(5);
        LBM_LOAD_F_ALIGNED(6);
        LBM_LOAD_F_ALIGNED(7);
        LBM_LOAD_F_ALIGNED(8);
        LBM_LOAD_F_ALIGNED(9);
        LBM_LOAD_F_ALIGNED(10);
        LBM_LOAD_F_ALIGNED(11);
        LBM_LOAD_F_ALIGNED(12);
        LBM_LOAD_F_ALIGNED(13);
        LBM_LOAD_F_ALIGNED(14);
        LBM_LOAD_F_ALIGNED(15);
        LBM_LOAD_F_ALIGNED(16);
        LBM_LOAD_F_ALIGNED(17);
        LBM_LOAD_F_ALIGNED(18);
        LBM_LOAD_F_ALIGNED(19);
        LBM_LOAD_F_ALIGNED(20);
        LBM_LOAD_F_ALIGNED(21);
        LBM_LOAD_F_ALIGNED(22);
        LBM_LOAD_F_ALIGNED(23);
        LBM_LOAD_F_ALIGNED(24);
        LBM_LOAD_F_ALIGNED(25);
        LBM_LOAD_F_ALIGNED(26);

        const std::uint8_t type = node_type[tid];
        if (type != kFluid) {
            LBM_STORE_F_ALIGNED(0);
            LBM_STORE_F_ALIGNED(1);
            LBM_STORE_F_ALIGNED(2);
            LBM_STORE_F_ALIGNED(3);
            LBM_STORE_F_ALIGNED(4);
            LBM_STORE_F_ALIGNED(5);
            LBM_STORE_F_ALIGNED(6);
            LBM_STORE_F_ALIGNED(7);
            LBM_STORE_F_ALIGNED(8);
            LBM_STORE_F_ALIGNED(9);
            LBM_STORE_F_ALIGNED(10);
            LBM_STORE_F_ALIGNED(11);
            LBM_STORE_F_ALIGNED(12);
            LBM_STORE_F_ALIGNED(13);
            LBM_STORE_F_ALIGNED(14);
            LBM_STORE_F_ALIGNED(15);
            LBM_STORE_F_ALIGNED(16);
            LBM_STORE_F_ALIGNED(17);
            LBM_STORE_F_ALIGNED(18);
            LBM_STORE_F_ALIGNED(19);
            LBM_STORE_F_ALIGNED(20);
            LBM_STORE_F_ALIGNED(21);
            LBM_STORE_F_ALIGNED(22);
            LBM_STORE_F_ALIGNED(23);
            LBM_STORE_F_ALIGNED(24);
            LBM_STORE_F_ALIGNED(25);
            LBM_STORE_F_ALIGNED(26);
            return;
        }

#if defined(LBM_BENCH_STORE_INPUT)
        // Memory-path benchmark: preserve the 27 DF writes but skip collision math.
#else
        const Real fx = (cfg.mode == StreamwiseMode::PeriodicBodyForce) ? cfg.body_force_x : 0.0;
        const Real fy = 0.0;
        const Real fz = 0.0;
        Real rho =
            f_0 + f_1 + f_2 + f_3 + f_4 + f_5 + f_6 + f_7 + f_8 +
            f_9 + f_10 + f_11 + f_12 + f_13 + f_14 + f_15 + f_16 + f_17 +
            f_18 + f_19 + f_20 + f_21 + f_22 + f_23 + f_24 + f_25 + f_26;
        rho = rho > 1.0e-20 ? rho : 1.0e-20;
        const Real invrho = 1.0 / rho;
        Real ux =
            f_1 - f_2 + f_7 - f_8 + f_9 - f_10 + f_11 - f_12 + f_13 - f_14 +
            f_19 - f_20 + f_21 - f_22 + f_23 - f_24 + f_25 - f_26;
        Real uy =
            f_3 - f_4 + f_7 - f_8 - f_9 + f_10 + f_15 - f_16 + f_17 - f_18 +
            f_19 - f_20 + f_21 - f_22 - f_23 + f_24 - f_25 + f_26;
        Real uz =
            f_5 - f_6 + f_11 - f_12 - f_13 + f_14 + f_15 - f_16 - f_17 + f_18 +
            f_19 - f_20 - f_21 + f_22 + f_23 - f_24 - f_25 + f_26;
        ux = (ux + 0.5 * fx) * invrho;
        uy = (uy + 0.5 * fy) * invrho;
        uz = (uz + 0.5 * fz) * invrho;
        const Real uu = ux * ux + uy * uy + uz * uz;
        LBM_BENCH_ACCUM_DECL;
        const Real omega = cfg.omega;
		f_20 += omega*(_feq_mmm(rho,ux,uy,uz,uu) - f_20);
		f_8 += omega*(_feq_mmz(rho,ux,uy,uz,uu) - f_8);
		f_22 += omega*(_feq_mmp(rho,ux,uy,uz,uu) - f_22);
		f_12 += omega*(_feq_mzm(rho,ux,uy,uz,uu) - f_12);
		f_2 += omega*(_feq_mzz(rho,ux,uy,uz,uu) - f_2);
		f_14 += omega*(_feq_mzp(rho,ux,uy,uz,uu) - f_14);
		f_24 += omega*(_feq_mpm(rho,ux,uy,uz,uu) - f_24);
		f_10 += omega*(_feq_mpz(rho,ux,uy,uz,uu) - f_10);
		f_26 += omega*(_feq_mpp(rho,ux,uy,uz,uu) - f_26);
		f_16 += omega*(_feq_zmm(rho,ux,uy,uz,uu) - f_16);
		f_4 += omega*(_feq_zmz(rho,ux,uy,uz,uu) - f_4);
		f_18 += omega*(_feq_zmp(rho,ux,uy,uz,uu) - f_18);
		f_6 += omega*(_feq_zzm(rho,ux,uy,uz,uu) - f_6);
		f_0 += omega*(_feq_zzz(rho,ux,uy,uz,uu) - f_0);
		f_5 += omega*(_feq_zzp(rho,ux,uy,uz,uu) - f_5);
		f_17 += omega*(_feq_zpm(rho,ux,uy,uz,uu) - f_17);
		f_3 += omega*(_feq_zpz(rho,ux,uy,uz,uu) - f_3);
		f_15 += omega*(_feq_zpp(rho,ux,uy,uz,uu) - f_15);
		f_25 += omega*(_feq_pmm(rho,ux,uy,uz,uu) - f_25);
		f_9 += omega*(_feq_pmz(rho,ux,uy,uz,uu) - f_9);
		f_23 += omega*(_feq_pmp(rho,ux,uy,uz,uu) - f_23);
		f_13 += omega*(_feq_pzm(rho,ux,uy,uz,uu) - f_13);
		f_1 += omega*(_feq_pzz(rho,ux,uy,uz,uu) - f_1);
		f_11 += omega*(_feq_pzp(rho,ux,uy,uz,uu) - f_11);
		f_21 += omega*(_feq_ppm(rho,ux,uy,uz,uu) - f_21);
		f_7 += omega*(_feq_ppz(rho,ux,uy,uz,uu) - f_7);
		f_19 += omega*(_feq_ppp(rho,ux,uy,uz,uu) - f_19);
#endif

#if !defined(LBM_BENCH_COMPUTE_ONLY)
        LBM_STORE_F_ALIGNED(0);
        LBM_STORE_F_ALIGNED(1);
        LBM_STORE_F_ALIGNED(2);
        LBM_STORE_F_ALIGNED(3);
        LBM_STORE_F_ALIGNED(4);
        LBM_STORE_F_ALIGNED(5);
        LBM_STORE_F_ALIGNED(6);
        LBM_STORE_F_ALIGNED(7);
        LBM_STORE_F_ALIGNED(8);
        LBM_STORE_F_ALIGNED(9);
        LBM_STORE_F_ALIGNED(10);
        LBM_STORE_F_ALIGNED(11);
        LBM_STORE_F_ALIGNED(12);
        LBM_STORE_F_ALIGNED(13);
        LBM_STORE_F_ALIGNED(14);
        LBM_STORE_F_ALIGNED(15);
        LBM_STORE_F_ALIGNED(16);
        LBM_STORE_F_ALIGNED(17);
        LBM_STORE_F_ALIGNED(18);
        LBM_STORE_F_ALIGNED(19);
        LBM_STORE_F_ALIGNED(20);
        LBM_STORE_F_ALIGNED(21);
        LBM_STORE_F_ALIGNED(22);
        LBM_STORE_F_ALIGNED(23);
        LBM_STORE_F_ALIGNED(24);
        LBM_STORE_F_ALIGNED(25);
        LBM_STORE_F_ALIGNED(26);
#else
        LBM_BENCH_FLUSH_ALIGNED();
#endif
    }
}


__global__ __launch_bounds__(kLaunchBounds) void recover_macros_kernel(
    StreamingView view,
    const std::uint8_t* LBM_RESTRICT node_type,
    SimulationConfig cfg,
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
    load_logical_cell(view, x, y, z, cfg.nx, cfg.ny, cfg.nz, populations);

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
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg) {
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    initialize_equilibrium_kernel<<<grid, block>>>(view, d_node_type, cfg);
    cuda_check(cudaGetLastError(), "launch initialize_equilibrium_kernel");
}

void launch_collide_and_stream(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg) {
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    if (view.offset != nullptr) {
        collide_and_stream_kernel<true><<<grid, block>>>(view, d_node_type, cfg);
    } else {
        collide_and_stream_kernel<false><<<grid, block>>>(view, d_node_type, cfg);
    }
    cuda_check(cudaGetLastError(), "launch collide_and_stream_kernel");
}

void launch_recover_macros(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    Real* d_rho,
    Real* d_ux,
    Real* d_uy,
    Real* d_uz) {
    dim3 grid{};
    dim3 block{};
    volume_launch_config(cfg, &grid, &block);
    recover_macros_kernel<<<grid, block>>>(view, d_node_type, cfg, d_rho, d_ux, d_uy, d_uz);
    cuda_check(cudaGetLastError(), "launch recover_macros_kernel");
}

}  // namespace lbm
