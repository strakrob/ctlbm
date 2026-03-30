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
    int y;
    int z;
    int q;

    if (cell >= state.cell_count) {
        return;
    }

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
    CUdevice device;
    CUmemAllocationProp prop;
    CUmemAccessDesc access;
    int device_ordinal;
    int vmm_supported;
    size_t reservation_bytes;
    size_t q;

    memset(state, 0, sizeof(*state));
    state->cell_count = cfg->nx * cfg->ny * cfg->nz;
    state->population_bytes = sizeof(Real) * (size_t)state->cell_count;

    lbm_driver_check(cuInit(0), "initialize driver api");
    lbm_check(cudaGetDevice(&device_ordinal), "get runtime device");
    lbm_driver_check(cuDeviceGet(&device, device_ordinal), "get driver device");
    lbm_driver_check(cuDeviceGetAttribute(&vmm_supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device), "query vmm support");
    if (vmm_supported == 0) {
        fprintf(stderr, "cuda vmm is not supported on this gpu\n");
        exit(1);
    }

    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;

    lbm_driver_check(cuMemGetAllocationGranularity(&state->granularity_bytes, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM), "query vmm granularity");
    state->allocation_bytes =
        ((state->population_bytes + state->granularity_bytes - 1) / state->granularity_bytes) * state->granularity_bytes;
    state->logical_cells = state->cell_count;
    state->use_offset_fallback = (state->allocation_bytes != state->population_bytes) ? 1 : 0;

    if (state->use_offset_fallback) {
        reservation_bytes = (size_t)LBM_Q * state->allocation_bytes;
    } else {
        reservation_bytes = 2 * (size_t)LBM_Q * state->population_bytes;
    }
    lbm_driver_check(cuMemAddressReserve(&state->reservation, reservation_bytes, 0, 0, 0), "reserve vmm space");

    for (q = 0; q < (size_t)LBM_Q; ++q) {
        const CUdeviceptr base_ptr = state->use_offset_fallback
            ? state->reservation + (CUdeviceptr)(q * state->allocation_bytes)
            : state->reservation + (CUdeviceptr)(2 * q * state->population_bytes);

        lbm_driver_check(cuMemCreate(&state->handles[q], state->allocation_bytes, &prop, 0), "create vmm allocation");
        lbm_driver_check(cuMemMap(base_ptr, state->allocation_bytes, 0, state->handles[q], 0), "map primary view");
        if (!state->use_offset_fallback) {
            lbm_driver_check(cuMemMap(base_ptr + state->population_bytes, state->population_bytes, 0, state->handles[q], 0), "map alias view");
        }

        state->base[q] = (Real*)base_ptr;
        state->current[q] = state->base[q];
        state->p[q] = state->current[q];
        state->h_offset[q] = 0;
    }

    memset(&access, 0, sizeof(access));
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device_ordinal;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    lbm_driver_check(cuMemSetAccess(state->reservation, reservation_bytes, &access, 1), "set vmm access");

    if (state->use_offset_fallback) {
        lbm_check(cudaMalloc((void**)&state->d_offset, sizeof(int) * (size_t)LBM_Q), "allocate offset table");
        lbm_check(cudaMemcpy(state->d_offset, state->h_offset, sizeof(int) * (size_t)LBM_Q, cudaMemcpyHostToDevice), "initialize offsets");
    }
}

void lbm_destroy_state(LBMState* state) {
    size_t q;
    size_t reservation_bytes;

    if (state->d_offset != NULL) {
        cudaFree(state->d_offset);
        state->d_offset = NULL;
    }

    if (state->reservation == 0 || state->allocation_bytes == 0) {
        return;
    }

    for (q = 0; q < (size_t)LBM_Q; ++q) {
        const CUdeviceptr base_ptr = state->use_offset_fallback
            ? state->reservation + (CUdeviceptr)(q * state->allocation_bytes)
            : state->reservation + (CUdeviceptr)(2 * q * state->population_bytes);
        lbm_driver_check(cuMemUnmap(base_ptr, state->allocation_bytes), "unmap primary view");
        if (!state->use_offset_fallback) {
            lbm_driver_check(cuMemUnmap(base_ptr + state->population_bytes, state->population_bytes), "unmap alias view");
        }
    }

    for (q = 0; q < (size_t)LBM_Q; ++q) {
        if (state->handles[q] != 0) {
            lbm_driver_check(cuMemRelease(state->handles[q]), "release vmm allocation");
        }
    }

    reservation_bytes = state->use_offset_fallback
        ? (size_t)LBM_Q * state->allocation_bytes
        : 2 * (size_t)LBM_Q * state->population_bytes;
    lbm_driver_check(cuMemAddressFree(state->reservation, reservation_bytes), "free vmm reservation");
    memset(state, 0, sizeof(*state));
}

void lbm_advance_streaming(const LBMConfig* cfg, LBMState* state) {
    int q;

    if (state->use_offset_fallback) {
        for (q = 0; q < LBM_Q; ++q) {
            state->h_offset[q] -= lbm_streaming_offset(q, cfg);
            state->h_offset[q] %= state->logical_cells;
            if (state->h_offset[q] < 0) {
                state->h_offset[q] += state->logical_cells;
            }
        }
        lbm_check(cudaMemcpy(state->d_offset, state->h_offset, sizeof(int) * (size_t)LBM_Q, cudaMemcpyHostToDevice), "update offsets");
        return;
    }

    for (q = 0; q < LBM_Q; ++q) {
        const ptrdiff_t shift = -(ptrdiff_t)lbm_streaming_offset(q, cfg);
        state->current[q] += shift;
        if (state->current[q] < state->base[q]) {
            state->current[q] += (ptrdiff_t)state->cell_count;
        } else if (state->current[q] + state->cell_count > state->base[q] + 2 * state->cell_count) {
            state->current[q] -= (ptrdiff_t)state->cell_count;
        }
        state->p[q] = state->current[q];
    }
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
