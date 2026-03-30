#include "lbm.cuh"

static void lbm_boundary_volume_launch(const LBMConfig* cfg, dim3* grid, dim3* block) {
    const int blocks = (cfg->nx * cfg->ny * cfg->nz + LBM_BLOCK_SIZE - 1) / LBM_BLOCK_SIZE;
    *block = dim3((unsigned int)LBM_BLOCK_SIZE, 1u, 1u);
    *grid = dim3((unsigned int)blocks, 1u, 1u);
}

static void lbm_yz_launch(const LBMConfig* cfg, dim3* grid, dim3* block) {
    const int yz_count = cfg->ny * cfg->nz;
    const int blocks = (yz_count + LBM_BLOCK_SIZE - 1) / LBM_BLOCK_SIZE;
    *block = dim3((unsigned int)LBM_BLOCK_SIZE, 1u, 1u);
    *grid = dim3((unsigned int)blocks, 1u, 1u);
}

__global__ static void lbm_apply_walls_kernel(LBMState state, const unsigned char* node_type, LBMConfig cfg) {
    const int cell = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    Real f[LBM_Q];
    Real reflected[LBM_Q];
    int q;

    if (cell >= state.cell_count) {
        return;
    }

    if (node_type[cell] != LBM_NODE_WALL) {
        return;
    }

    lbm_load_cell(state, cell, f);

    #pragma unroll
    for (q = 0; q < LBM_Q; ++q) {
        reflected[q] = f[lbm_opp_value(q)];
    }

    lbm_store_cell(state, cell, reflected);
}

__global__ static void lbm_apply_mode_c_boundaries_kernel(LBMState state, const unsigned char* node_type, LBMConfig cfg) {
    const int yz = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int y;
    int z;
    int inlet_cell;
    int outlet_cell;
    int q;

    if (yz >= cfg.ny * cfg.nz) {
        return;
    }

    y = yz % cfg.ny;
    z = yz / cfg.ny;

    if (y == 0 || y == cfg.ny - 1 || z == 0 || z == cfg.nz - 1) {
        return;
    }

    inlet_cell = lbm_cell_index(0, y, z, cfg.nx, cfg.ny);
    outlet_cell = lbm_cell_index(cfg.nx - 1, y, z, cfg.nx, cfg.ny);

    if (node_type[inlet_cell] == LBM_NODE_INLET) {
        Real interior[LBM_Q];
        Real rho_i;
        Real ux_i;
        Real uy_i;
        Real uz_i;
        const Real ux_b = lbm_inlet_velocity(y, z, &cfg);

        lbm_load_cell(state, lbm_cell_index(1, y, z, cfg.nx, cfg.ny), interior);
        lbm_recover_macro(interior, &rho_i, &ux_i, &uy_i, &uz_i);

        #pragma unroll
        for (q = 0; q < LBM_Q; ++q) {
            const Real feq_b = lbm_equilibrium(q, rho_i, ux_b, 0.0, 0.0);
            const Real feq_i = lbm_equilibrium(q, rho_i, ux_i, uy_i, uz_i);
            lbm_store_population(state, q, 0, y, z, cfg, feq_b + (interior[q] - feq_i));
        }
    }

    if (node_type[outlet_cell] == LBM_NODE_OUTLET) {
        Real interior[LBM_Q];

        lbm_load_cell(state, lbm_cell_index(cfg.nx - 2, y, z, cfg.nx, cfg.ny), interior);

        if (cfg.outlet_mode == LBM_OUTLET_EXTRAPOLATION) {
            #pragma unroll
            for (q = 0; q < LBM_Q; ++q) {
                lbm_store_population(state, q, cfg.nx - 1, y, z, cfg, interior[q]);
            }
        } else {
            Real rho_i;
            Real ux_i;
            Real uy_i;
            Real uz_i;

            lbm_recover_macro(interior, &rho_i, &ux_i, &uy_i, &uz_i);

            #pragma unroll
            for (q = 0; q < LBM_Q; ++q) {
                const Real feq_b = lbm_equilibrium(q, cfg.rho0, ux_i, uy_i, uz_i);
                const Real feq_i = lbm_equilibrium(q, rho_i, ux_i, uy_i, uz_i);
                lbm_store_population(state, q, cfg.nx - 1, y, z, cfg, feq_b + (interior[q] - feq_i));
            }
        }
    }
}

void lbm_launch_apply_walls(LBMState state, const unsigned char* d_node_type, LBMConfig cfg) {
    dim3 grid;
    dim3 block;

    lbm_boundary_volume_launch(&cfg, &grid, &block);
    lbm_apply_walls_kernel<<<grid, block>>>(state, d_node_type, cfg);
    lbm_check(cudaGetLastError(), "launch walls");
}

void lbm_launch_apply_mode_c_boundaries(LBMState state, const unsigned char* d_node_type, LBMConfig cfg) {
    dim3 grid;
    dim3 block;

    lbm_yz_launch(&cfg, &grid, &block);
    lbm_apply_mode_c_boundaries_kernel<<<grid, block>>>(state, d_node_type, cfg);
    lbm_check(cudaGetLastError(), "launch mode C boundaries");
}
