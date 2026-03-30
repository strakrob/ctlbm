#include "lbm.cuh"

static void lbm_volume_launch(const LBMConfig* cfg, dim3* grid, dim3* block) {
    const int blocks = (cfg->nx * cfg->ny * cfg->nz + LBM_BLOCK_SIZE - 1) / LBM_BLOCK_SIZE;
    *block = dim3((unsigned int)LBM_BLOCK_SIZE, 1u, 1u);
    *grid = dim3((unsigned int)blocks, 1u, 1u);
}

__device__ static LBM_INLINE int lbm_thread_cell(const LBMConfig cfg) {
    return (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
}

__global__ static void lbm_classify_nodes_kernel(unsigned char* node_type, LBMConfig cfg) {
    const int cell = lbm_thread_cell(cfg);
    int x;
    int y;
    int z;

    if (cell >= cfg.nx * cfg.ny * cfg.nz) {
        return;
    }

    x = cell % cfg.nx;
    y = (cell / cfg.nx) % cfg.ny;
    z = cell / (cfg.nx * cfg.ny);

    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        node_type[cell] = LBM_NODE_WALL;
    } else if (x == 0) {
        node_type[cell] = LBM_NODE_INLET;
    } else if (x == cfg.nx - 1) {
        node_type[cell] = LBM_NODE_OUTLET;
    } else {
        node_type[cell] = LBM_NODE_FLUID;
    }
}

__global__ static void lbm_initialize_kernel(LBMState state, const unsigned char* node_type, LBMConfig cfg) {
    const int cell = lbm_thread_cell(cfg);
    Real f[LBM_Q];
    Real rho;
    Real ux;
    int x;
    int y;
    int z;
    int q;

    if (cell >= state.cell_count) {
        return;
    }

    x = cell % cfg.nx;
    y = (cell / cfg.nx) % cfg.ny;
    z = cell / (cfg.nx * cfg.ny);

    rho = cfg.rho0;
    ux = 0.0;

    if (node_type[cell] == LBM_NODE_INLET) {
        ux = lbm_inlet_velocity(y, z, &cfg);
    }

    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        f[q] = lbm_equilibrium(q, rho, ux, 0.0, 0.0);
    }

    lbm_store_cell(state, cell, f);
}

__global__ static void lbm_collide_and_stream_kernel(LBMState state, const unsigned char* node_type, LBMConfig cfg) {
    const int cell = lbm_thread_cell(cfg);
    Real f[LBM_Q];
    Real rho;
    Real ux;
    Real uy;
    Real uz;
    int q;

    if (cell >= state.cell_count) {
        return;
    }

    lbm_load_cell(state, cell, f);

    if (node_type[cell] != LBM_NODE_FLUID) {
        lbm_store_cell(state, cell, f);
        return;
    }

    lbm_recover_macro(f, &rho, &ux, &uy, &uz);

    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        const Real feq = lbm_equilibrium(q, rho, ux, uy, uz);
        f[q] = f[q] - cfg.omega * (f[q] - feq);
    }

    lbm_store_cell(state, cell, f);
}

void lbm_copy_constants(void) {
}

void lbm_create_state(const LBMConfig* cfg, LBMState* state) {
    int q;

    state->cell_count = cfg->nx * cfg->ny * cfg->nz;
    for (q = 0; q < LBM_Q; ++q) {
        state->h_offset[q] = 0;
    }

    lbm_check(cudaMalloc((void**)&state->f, sizeof(Real) * (size_t)LBM_Q * (size_t)state->cell_count), "allocate distributions");
    lbm_check(cudaMalloc((void**)&state->d_offset, sizeof(int) * (size_t)LBM_Q), "allocate offsets");
    lbm_check(cudaMemcpy(state->d_offset, state->h_offset, sizeof(int) * (size_t)LBM_Q, cudaMemcpyHostToDevice), "initialize offsets");
}

void lbm_destroy_state(LBMState* state) {
    if (state->d_offset != NULL) {
        cudaFree(state->d_offset);
        state->d_offset = NULL;
    }
    if (state->f != NULL) {
        cudaFree(state->f);
        state->f = NULL;
    }
    state->cell_count = 0;
}

void lbm_advance_offsets(const LBMConfig* cfg, LBMState* state) {
    int q;

    for (q = 0; q < LBM_Q; ++q) {
        state->h_offset[q] = lbm_wrap(state->h_offset[q] - lbm_streaming_offset(q, cfg), state->cell_count);
    }

    lbm_check(cudaMemcpy(state->d_offset, state->h_offset, sizeof(int) * (size_t)LBM_Q, cudaMemcpyHostToDevice), "update offsets");
}

void lbm_launch_classify_nodes(unsigned char* d_node_type, LBMConfig cfg) {
    dim3 grid;
    dim3 block;

    lbm_volume_launch(&cfg, &grid, &block);
    lbm_classify_nodes_kernel<<<grid, block>>>(d_node_type, cfg);
    lbm_check(cudaGetLastError(), "launch classify");
}

void lbm_launch_initialize(LBMState state, const unsigned char* d_node_type, LBMConfig cfg) {
    dim3 grid;
    dim3 block;

    lbm_volume_launch(&cfg, &grid, &block);
    lbm_initialize_kernel<<<grid, block>>>(state, d_node_type, cfg);
    lbm_check(cudaGetLastError(), "launch initialize");
}

void lbm_launch_collide_and_stream(LBMState state, const unsigned char* d_node_type, LBMConfig cfg) {
    dim3 grid;
    dim3 block;

    lbm_volume_launch(&cfg, &grid, &block);
    lbm_collide_and_stream_kernel<<<grid, block>>>(state, d_node_type, cfg);
    lbm_check(cudaGetLastError(), "launch collide");
}

#include "boundary_conditions.cu"
