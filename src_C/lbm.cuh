#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef LBM_USE_FLOAT
typedef float Real;
#else
typedef double Real;
#endif

enum {
    LBM_Q = 27,
    LBM_NODE_FLUID = 0,
    LBM_NODE_WALL = 1,
    LBM_NODE_INLET = 2,
    LBM_NODE_OUTLET = 3,
    LBM_INLET_UNIFORM = 0,
    LBM_INLET_PARABOLIC = 1,
    LBM_OUTLET_EXTRAPOLATION = 0,
    LBM_OUTLET_ZERO_PRESSURE = 1
};

#ifdef LBM_USE_FLOAT
#define LBM_BLOCK_SIZE 64
#else
#define LBM_BLOCK_SIZE 128
#endif

#ifdef __CUDACC__
#define LBM_INLINE __forceinline__
#else
#define LBM_INLINE inline
#endif

typedef struct {
    int nx;
    int ny;
    int nz;
    int steps;
    int inlet_profile;
    int outlet_mode;
    Real tau;
    Real omega;
    Real rho0;
    Real inlet_velocity;
} LBMConfig;

typedef struct {
    Real* f;
    int* d_offset;
    int h_offset[LBM_Q];
    int cell_count;
} LBMState;

static const int lbm_cx[LBM_Q] = {
    0,
    1, -1, 0, 0, 0, 0,
    1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0,
    1, -1, 1, -1, 1, -1, 1, -1
};

static const int lbm_cy[LBM_Q] = {
    0,
    0, 0, 1, -1, 0, 0,
    1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1,
    1, -1, 1, -1, -1, 1, -1, 1
};

static const int lbm_cz[LBM_Q] = {
    0,
    0, 0, 0, 0, 1, -1,
    0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1,
    1, -1, -1, 1, 1, -1, -1, 1
};

static const int lbm_opp[LBM_Q] = {
    0,
    2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 24, 23, 26, 25
};

static const Real lbm_w[LBM_Q] = {
    (Real)(8.0 / 27.0),
    (Real)(2.0 / 27.0), (Real)(2.0 / 27.0), (Real)(2.0 / 27.0), (Real)(2.0 / 27.0), (Real)(2.0 / 27.0), (Real)(2.0 / 27.0),
    (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0),
    (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0), (Real)(1.0 / 54.0),
    (Real)(1.0 / 216.0), (Real)(1.0 / 216.0), (Real)(1.0 / 216.0), (Real)(1.0 / 216.0),
    (Real)(1.0 / 216.0), (Real)(1.0 / 216.0), (Real)(1.0 / 216.0), (Real)(1.0 / 216.0)
};

__host__ __device__ static LBM_INLINE int lbm_cx_value(int q) {
    switch (q) {
        case 0: return 0;
        case 1: return 1; case 2: return -1; case 3: return 0; case 4: return 0; case 5: return 0; case 6: return 0;
        case 7: return 1; case 8: return -1; case 9: return 1; case 10: return -1; case 11: return 1; case 12: return -1;
        case 13: return 1; case 14: return -1; case 15: return 0; case 16: return 0; case 17: return 0; case 18: return 0;
        case 19: return 1; case 20: return -1; case 21: return 1; case 22: return -1; case 23: return 1; case 24: return -1;
        case 25: return 1; case 26: return -1;
        default: return 0;
    }
}

__host__ __device__ static LBM_INLINE int lbm_cy_value(int q) {
    switch (q) {
        case 0: return 0;
        case 1: return 0; case 2: return 0; case 3: return 1; case 4: return -1; case 5: return 0; case 6: return 0;
        case 7: return 1; case 8: return -1; case 9: return -1; case 10: return 1; case 11: return 0; case 12: return 0;
        case 13: return 0; case 14: return 0; case 15: return 1; case 16: return -1; case 17: return 1; case 18: return -1;
        case 19: return 1; case 20: return -1; case 21: return 1; case 22: return -1; case 23: return -1; case 24: return 1;
        case 25: return -1; case 26: return 1;
        default: return 0;
    }
}

__host__ __device__ static LBM_INLINE int lbm_cz_value(int q) {
    switch (q) {
        case 0: return 0;
        case 1: return 0; case 2: return 0; case 3: return 0; case 4: return 0; case 5: return 1; case 6: return -1;
        case 7: return 0; case 8: return 0; case 9: return 0; case 10: return 0; case 11: return 1; case 12: return -1;
        case 13: return -1; case 14: return 1; case 15: return 1; case 16: return -1; case 17: return -1; case 18: return 1;
        case 19: return 1; case 20: return -1; case 21: return -1; case 22: return 1; case 23: return 1; case 24: return -1;
        case 25: return -1; case 26: return 1;
        default: return 0;
    }
}

__host__ __device__ static LBM_INLINE int lbm_opp_value(int q) {
    switch (q) {
        case 0: return 0;
        case 1: return 2; case 2: return 1; case 3: return 4; case 4: return 3; case 5: return 6; case 6: return 5;
        case 7: return 8; case 8: return 7; case 9: return 10; case 10: return 9; case 11: return 12; case 12: return 11;
        case 13: return 14; case 14: return 13; case 15: return 16; case 16: return 15; case 17: return 18; case 18: return 17;
        case 19: return 20; case 20: return 19; case 21: return 22; case 22: return 21; case 23: return 24; case 24: return 23;
        case 25: return 26; case 26: return 25;
        default: return 0;
    }
}

__host__ __device__ static LBM_INLINE Real lbm_weight_value(int q) {
    switch (q) {
        case 0: return (Real)(8.0 / 27.0);
        case 1: case 2: case 3: case 4: case 5: case 6: return (Real)(2.0 / 27.0);
        case 7: case 8: case 9: case 10: case 11: case 12:
        case 13: case 14: case 15: case 16: case 17: case 18: return (Real)(1.0 / 54.0);
        default: return (Real)(1.0 / 216.0);
    }
}

__host__ __device__ static LBM_INLINE int lbm_cell_index(int x, int y, int z, int nx, int ny) {
    return ((z * ny) + y) * nx + x;
}

static LBM_INLINE int lbm_streaming_offset(int q, const LBMConfig* cfg) {
    return lbm_cx[q] + lbm_cy[q] * cfg->nx + lbm_cz[q] * cfg->nx * cfg->ny;
}

static LBM_INLINE int lbm_wrap(int value, int period) {
    value %= period;
    if (value < 0) {
        value += period;
    }
    return value;
}

__device__ static LBM_INLINE int lbm_physical_index(const LBMState state, int q, int cell) {
    const int shifted = cell + state.d_offset[q];
    const int wrapped = shifted < state.cell_count ? shifted : shifted - state.cell_count;
    return q * state.cell_count + wrapped;
}

__device__ static LBM_INLINE void lbm_load_cell(const LBMState state, int cell, Real f[LBM_Q]) {
    int q;
    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        f[q] = state.f[lbm_physical_index(state, q, cell)];
    }
}

__device__ static LBM_INLINE void lbm_store_cell(const LBMState state, int cell, const Real f[LBM_Q]) {
    int q;
    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        state.f[lbm_physical_index(state, q, cell)] = f[q];
    }
}

__device__ static LBM_INLINE void lbm_store_population(
    const LBMState state,
    int q,
    int x,
    int y,
    int z,
    const LBMConfig cfg,
    Real value) {
    const int cell = lbm_cell_index(x, y, z, cfg.nx, cfg.ny);
    state.f[lbm_physical_index(state, q, cell)] = value;
}

__device__ static LBM_INLINE Real lbm_equilibrium(int q, Real rho, Real ux, Real uy, Real uz) {
    const Real cu = (Real)lbm_cx_value(q) * ux + (Real)lbm_cy_value(q) * uy + (Real)lbm_cz_value(q) * uz;
    const Real uu = ux * ux + uy * uy + uz * uz;
    return lbm_weight_value(q) * rho * ((Real)1.0 + (Real)3.0 * cu + (Real)4.5 * cu * cu - (Real)1.5 * uu);
}

__device__ static LBM_INLINE void lbm_recover_macro(
    const Real f[LBM_Q],
    Real* rho,
    Real* ux,
    Real* uy,
    Real* uz) {
    Real density = 0.0;
    Real mx = 0.0;
    Real my = 0.0;
    Real mz = 0.0;
    int q;

    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        density += f[q];
        mx += f[q] * (Real)lbm_cx_value(q);
        my += f[q] * (Real)lbm_cy_value(q);
        mz += f[q] * (Real)lbm_cz_value(q);
    }

    if (density < (Real)1.0e-20) {
        density = (Real)1.0e-20;
    }

    *rho = density;
    *ux = mx / density;
    *uy = my / density;
    *uz = mz / density;
}

__host__ __device__ static LBM_INLINE Real lbm_inlet_velocity(int y, int z, const LBMConfig* cfg) {
    if (cfg->inlet_profile == LBM_INLET_UNIFORM) {
        return cfg->inlet_velocity;
    }

    {
        const Real yn = (Real)y / (Real)(cfg->ny - 1);
        const Real zn = (Real)z / (Real)(cfg->nz - 1);
        return (Real)16.0 * cfg->inlet_velocity * yn * ((Real)1.0 - yn) * zn * ((Real)1.0 - zn);
    }
}

static LBM_INLINE void lbm_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        exit(1);
    }
}

void lbm_copy_constants(void);
void lbm_create_state(const LBMConfig* cfg, LBMState* state);
void lbm_destroy_state(LBMState* state);
void lbm_advance_offsets(const LBMConfig* cfg, LBMState* state);

void lbm_launch_classify_nodes(unsigned char* d_node_type, LBMConfig cfg);
void lbm_launch_initialize(LBMState state, const unsigned char* d_node_type, LBMConfig cfg);
void lbm_launch_collide_and_stream(LBMState state, const unsigned char* d_node_type, LBMConfig cfg);
void lbm_launch_apply_walls(LBMState state, const unsigned char* d_node_type, LBMConfig cfg);
void lbm_launch_apply_mode_c_boundaries(LBMState state, const unsigned char* d_node_type, LBMConfig cfg);
