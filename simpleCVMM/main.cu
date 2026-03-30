#include "lbm.cuh"

#include <string.h>

static void lbm_usage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("  --nx N --ny N --nz N\n");
    printf("  --steps N --tau VALUE --rho0 VALUE\n");
    printf("  --inlet-velocity VALUE\n");
    printf("  --inlet-profile uniform|parabolic\n");
    printf("  --outlet extrapolation|zero-gauge-pressure\n");
    exit(0);
}

static void lbm_set_defaults(LBMConfig* cfg) {
    cfg->nx = 128;
    cfg->ny = 64;
    cfg->nz = 32;
    cfg->steps = 2000;
    cfg->inlet_profile = LBM_INLET_PARABOLIC;
    cfg->outlet_mode = LBM_OUTLET_EXTRAPOLATION;
    cfg->tau = (Real)0.8;
    cfg->omega = (Real)1.0 / cfg->tau;
    cfg->rho0 = (Real)1.0;
    cfg->inlet_velocity = (Real)0.02;
}

static int lbm_read_int(int argc, char** argv, int* i) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", argv[*i]);
        exit(1);
    }
    *i += 1;
    return atoi(argv[*i]);
}

static Real lbm_read_real(int argc, char** argv, int* i) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", argv[*i]);
        exit(1);
    }
    *i += 1;
    return (Real)atof(argv[*i]);
}

static void lbm_parse_args(int argc, char** argv, LBMConfig* cfg) {
    int i;

    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            lbm_usage(argv[0]);
        } else if (strcmp(argv[i], "--nx") == 0) {
            cfg->nx = lbm_read_int(argc, argv, &i);
        } else if (strcmp(argv[i], "--ny") == 0) {
            cfg->ny = lbm_read_int(argc, argv, &i);
        } else if (strcmp(argv[i], "--nz") == 0) {
            cfg->nz = lbm_read_int(argc, argv, &i);
        } else if (strcmp(argv[i], "--steps") == 0) {
            cfg->steps = lbm_read_int(argc, argv, &i);
        } else if (strcmp(argv[i], "--tau") == 0) {
            cfg->tau = lbm_read_real(argc, argv, &i);
        } else if (strcmp(argv[i], "--rho0") == 0) {
            cfg->rho0 = lbm_read_real(argc, argv, &i);
        } else if (strcmp(argv[i], "--inlet-velocity") == 0) {
            cfg->inlet_velocity = lbm_read_real(argc, argv, &i);
        } else if (strcmp(argv[i], "--inlet-profile") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --inlet-profile\n");
                exit(1);
            }
            i += 1;
            if (strcmp(argv[i], "uniform") == 0) {
                cfg->inlet_profile = LBM_INLET_UNIFORM;
            } else if (strcmp(argv[i], "parabolic") == 0) {
                cfg->inlet_profile = LBM_INLET_PARABOLIC;
            } else {
                fprintf(stderr, "unknown inlet profile: %s\n", argv[i]);
                exit(1);
            }
        } else if (strcmp(argv[i], "--outlet") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --outlet\n");
                exit(1);
            }
            i += 1;
            if (strcmp(argv[i], "extrapolation") == 0) {
                cfg->outlet_mode = LBM_OUTLET_EXTRAPOLATION;
            } else if (strcmp(argv[i], "zero-gauge-pressure") == 0) {
                cfg->outlet_mode = LBM_OUTLET_ZERO_PRESSURE;
            } else {
                fprintf(stderr, "unknown outlet mode: %s\n", argv[i]);
                exit(1);
            }
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            exit(1);
        }
    }
}

static void lbm_validate(const LBMConfig* cfg) {
    if (cfg->nx < 4 || cfg->ny < 4 || cfg->nz < 4) {
        fprintf(stderr, "nx, ny and nz must be at least 4\n");
        exit(1);
    }
    if (cfg->tau <= (Real)0.5) {
        fprintf(stderr, "tau must be greater than 0.5\n");
        exit(1);
    }
}

int main(int argc, char** argv) {
    LBMConfig cfg;
    LBMState state;
    unsigned char* d_node_type;
    int step;

    lbm_set_defaults(&cfg);
    lbm_parse_args(argc, argv, &cfg);
    cfg.omega = (Real)1.0 / cfg.tau;
    lbm_validate(&cfg);

    memset(&state, 0, sizeof(state));
    d_node_type = NULL;

    lbm_copy_constants();
    lbm_create_state(&cfg, &state);
    lbm_check(cudaMalloc((void**)&d_node_type, (size_t)state.cell_count * sizeof(unsigned char)), "allocate node types");

    lbm_launch_classify_nodes(d_node_type, cfg);
    lbm_launch_initialize(state, d_node_type, cfg);
    lbm_check(cudaDeviceSynchronize(), "initialize solver");

    for (step = 0; step < cfg.steps; ++step) {
        lbm_launch_collide_and_stream(state, d_node_type, cfg);
        lbm_advance_streaming(&cfg, &state);
        lbm_launch_apply_walls(state, d_node_type, cfg);
        lbm_launch_apply_mode_c_boundaries(state, d_node_type, cfg);
    }

    lbm_check(cudaDeviceSynchronize(), "run solver");
    printf("completed %d steps with vmm streaming\n", cfg.steps);

    if (d_node_type != NULL) {
        cudaFree(d_node_type);
    }
    lbm_destroy_state(&state);
    return 0;
}
