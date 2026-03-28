#include "lbm.cuh"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using lbm::InletProfile;
using lbm::NodeType;
using lbm::OutletMode;
using lbm::Real;
using lbm::SimulationConfig;
using lbm::StreamwiseMode;

struct RuntimeOptions {
    std::string output_dir = ".";
    std::string diagnostics_csv = "diagnostics.csv";
    bool write_cross_sections = false;
    bool do_not_write_full_volume = false;
    bool write_node_map = false;
};

struct Diagnostics {
    int step = 0;
    Real total_mass = Real(0.0);
    Real mean_density = Real(0.0);
    Real bulk_velocity = Real(0.0);
    Real flow_rate = Real(0.0);
    Real max_streamwise_velocity = Real(0.0);
    Real residual = Real(0.0);
    Real l2_error = Real(0.0);
    Real balance_metric = Real(0.0);
    Real mlups_current = Real(0.0);
    Real mlups_min = Real(0.0);
    Real mlups_avg = Real(0.0);
    Real mlups_max = Real(0.0);
};

struct MlupsStats {
    using Clock = std::chrono::steady_clock;

    Clock::time_point last_sample_time{};
    int last_step = 0;
    double min_mlups = std::numeric_limits<double>::infinity();
    double max_mlups = 0.0;
    double total_updates = 0.0;
    double total_seconds = 0.0;
    double current_mlups = 0.0;
    bool armed = false;
};

struct MemoryReport {
    std::size_t host_bytes = 0;
    std::size_t gpu_bytes = 0;
    std::size_t total_bytes = 0;
    double host_bytes_per_cell = 0.0;
    double gpu_bytes_per_cell = 0.0;
    double total_bytes_per_cell = 0.0;
};

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string mode_name(StreamwiseMode mode) {
    switch (mode) {
        case StreamwiseMode::PeriodicBodyForce:
            return "A";
        case StreamwiseMode::Pressure:
            return "B";
        case StreamwiseMode::Velocity:
            return "C";
    }
    return "?";
}

std::string outlet_name(OutletMode outlet_mode) {
    return outlet_mode == OutletMode::Extrapolation ? "extrapolation" : "zero-gauge-pressure";
}

std::string profile_name(InletProfile profile) {
    return profile == InletProfile::Uniform ? "uniform" : "parabolic";
}

std::string format_bytes(std::size_t bytes) {
    std::ostringstream os;
    os << bytes << " B (" << std::fixed << std::setprecision(3) << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MiB)";
    return os.str();
}

[[noreturn]] void usage_and_exit(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --nx N --ny N --nz N\n"
        << "  --steps N --output-every N --diag-every N\n"
        << "  --tau VALUE --mode A|B|C\n"
        << "  --force-x VALUE\n"
        << "  --rho0 VALUE --rho-inlet VALUE --rho-outlet VALUE\n"
        << "  --inlet-velocity VALUE --inlet-profile uniform|parabolic\n"
        << "  --outlet extrapolation|zero-gauge-pressure\n"
        << "  --write-cross-sections\n"
        << "  --write-node-map\n"
        << "  --do-not-write-full-volume\n"
        << "  --series-terms N --output-dir PATH\n";
    std::exit(0);
}

template <typename T>
T parse_number(const std::string& text, const char* what) {
    std::istringstream is(text);
    T value{};
    is >> value;
    if (!is || !is.eof()) {
        throw std::runtime_error(std::string("Unable to parse ") + what + " from '" + text + "'");
    }
    return value;
}

void parse_arguments(int argc, char** argv, SimulationConfig* cfg, RuntimeOptions* runtime) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto require_value = [&](const char* option_name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + option_name);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            usage_and_exit(argv[0]);
        } else if (arg == "--nx") {
            cfg->nx = parse_number<int>(require_value("--nx"), "nx");
        } else if (arg == "--ny") {
            cfg->ny = parse_number<int>(require_value("--ny"), "ny");
        } else if (arg == "--nz") {
            cfg->nz = parse_number<int>(require_value("--nz"), "nz");
        } else if (arg == "--steps") {
            cfg->nsteps = parse_number<int>(require_value("--steps"), "step count");
        } else if (arg == "--output-every") {
            cfg->output_every = parse_number<int>(require_value("--output-every"), "output interval");
        } else if (arg == "--diag-every") {
            cfg->diagnostic_every = parse_number<int>(require_value("--diag-every"), "diagnostic interval");
        } else if (arg == "--tau") {
            cfg->tau = parse_number<Real>(require_value("--tau"), "tau");
        } else if (arg == "--force-x") {
            cfg->body_force_x = parse_number<Real>(require_value("--force-x"), "body force");
        } else if (arg == "--rho0") {
            cfg->rho0 = parse_number<Real>(require_value("--rho0"), "reference density");
        } else if (arg == "--rho-inlet") {
            cfg->rho_inlet = parse_number<Real>(require_value("--rho-inlet"), "inlet density");
        } else if (arg == "--rho-outlet") {
            cfg->rho_outlet = parse_number<Real>(require_value("--rho-outlet"), "outlet density");
        } else if (arg == "--inlet-velocity") {
            cfg->inlet_velocity = parse_number<Real>(require_value("--inlet-velocity"), "inlet velocity");
        } else if (arg == "--series-terms") {
            cfg->analytical_terms = parse_number<int>(require_value("--series-terms"), "series term count");
        } else if (arg == "--output-dir") {
            runtime->output_dir = require_value("--output-dir");
        } else if (arg == "--write-cross-sections") {
            runtime->write_cross_sections = true;
        } else if (arg == "--write-node-map") {
            runtime->write_node_map = true;
        } else if (arg == "--do-not-write-full-volume") {
            runtime->do_not_write_full_volume = true;
        } else if (arg == "--mode") {
            const std::string mode = to_lower(require_value("--mode"));
            if (mode == "a" || mode == "periodic" || mode == "body-force") {
                cfg->mode = StreamwiseMode::PeriodicBodyForce;
            } else if (mode == "b" || mode == "pressure") {
                cfg->mode = StreamwiseMode::Pressure;
            } else if (mode == "c" || mode == "velocity") {
                cfg->mode = StreamwiseMode::Velocity;
            } else {
                throw std::runtime_error("Unsupported mode '" + mode + "'");
            }
        } else if (arg == "--outlet") {
            const std::string outlet = to_lower(require_value("--outlet"));
            if (outlet == "extrapolation" || outlet == "copy") {
                cfg->outlet_mode = OutletMode::Extrapolation;
            } else if (outlet == "zero-gauge-pressure" || outlet == "pressure") {
                cfg->outlet_mode = OutletMode::ZeroGaugePressure;
            } else {
                throw std::runtime_error("Unsupported outlet mode '" + outlet + "'");
            }
        } else if (arg == "--inlet-profile") {
            const std::string profile = to_lower(require_value("--inlet-profile"));
            if (profile == "uniform") {
                cfg->inlet_profile = InletProfile::Uniform;
            } else if (profile == "parabolic") {
                cfg->inlet_profile = InletProfile::Parabolic;
            } else {
                throw std::runtime_error("Unsupported inlet profile '" + profile + "'");
            }
        } else {
            throw std::runtime_error("Unknown argument '" + arg + "'");
        }
    }

    runtime->diagnostics_csv = runtime->output_dir + "/diagnostics.csv";
}

void validate_config(SimulationConfig* cfg) {
    if (cfg->nx < 4 || cfg->ny < 4 || cfg->nz < 4) {
        throw std::runtime_error("The duct needs at least 4 nodes in every direction.");
    }
    if (cfg->tau <= Real(0.5)) {
        throw std::runtime_error("tau must be greater than 0.5 for positive viscosity.");
    }
    if (cfg->analytical_terms < 1) {
        throw std::runtime_error("series-terms must be positive.");
    }
    cfg->omega = Real(1.0) / cfg->tau;
}

Real plane_flow_rate(const SimulationConfig& cfg, const std::vector<std::uint8_t>& node_type, const std::vector<Real>& ux, int x_plane) {
    Real sum = Real(0.0);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            const int cell = lbm::flatten_xyz(x_plane, y, z, cfg.nx, cfg.ny, cfg.nz);
            if (node_type[cell] != lbm::kWall) {
                sum += ux[cell];
            }
        }
    }
    return sum;
}

Real rectangular_duct_velocity(Real forcing, Real viscosity, int y, int z, const SimulationConfig& cfg) {
    if (forcing == Real(0.0)) {
        return Real(0.0);
    }

    const Real pi = Real(3.1415926535897932384626433832795);
    const Real h = Real(cfg.ny - 1);
    const Real w = Real(cfg.nz - 1);
    const Real yy = Real(y);
    const Real zz = Real(z);

    Real sum = Real(0.0);
    for (int n = 1, used = 0; used < cfg.analytical_terms; n += 2, ++used) {
        const Real nn = Real(n);
        const Real alpha = nn * pi / h;
        const Real cosine_ratio = std::cosh(alpha * (zz - Real(0.5) * w)) / std::cosh(alpha * Real(0.5) * w);
        sum += (Real(1.0) / (nn * nn * nn)) * (Real(1.0) - cosine_ratio) * std::sin(alpha * yy);
    }

    return Real(4.0) * forcing * h * h * sum / (viscosity * pi * pi * pi);
}

Diagnostics compute_diagnostics(
    const SimulationConfig& cfg,
    int step,
    const std::vector<std::uint8_t>& node_type,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    Real initial_mass,
    Real previous_bulk_velocity) {
    Diagnostics diagnostics;
    diagnostics.step = step;

    const int cell_count = cfg.nx * cfg.ny * cfg.nz;
    int fluid_nodes = 0;
    Real total_mass = Real(0.0);
    Real max_ux = -std::numeric_limits<Real>::max();

    for (int cell = 0; cell < cell_count; ++cell) {
        total_mass += rho[cell];
        if (node_type[cell] != lbm::kWall) {
            max_ux = std::max(max_ux, ux[cell]);
            ++fluid_nodes;
        }
    }

    const int interior_x = std::min(std::max(1, cfg.nx / 2), cfg.nx - 2);
    const Real flow_rate = plane_flow_rate(cfg, node_type, ux, interior_x);
    const Real bulk_velocity = flow_rate / Real(lbm::interior_area(cfg));

    diagnostics.total_mass = total_mass;
    diagnostics.mean_density = total_mass / Real(cell_count);
    diagnostics.flow_rate = flow_rate;
    diagnostics.bulk_velocity = bulk_velocity;
    diagnostics.max_streamwise_velocity = (fluid_nodes > 0) ? max_ux : Real(0.0);
    diagnostics.residual = (step == 0) ? Real(0.0) : std::abs(bulk_velocity - previous_bulk_velocity);

    if (step == 0) {
        diagnostics.balance_metric = Real(0.0);
    } else if (cfg.mode == StreamwiseMode::PeriodicBodyForce) {
        diagnostics.balance_metric = std::abs(total_mass - initial_mass) / std::max(std::abs(initial_mass), Real(1.0e-20));
    } else {
        const Real q_in = plane_flow_rate(cfg, node_type, ux, 0);
        const Real q_out = plane_flow_rate(cfg, node_type, ux, cfg.nx - 1);
        const Real reference_flux = Real(0.5) * (std::abs(q_in) + std::abs(q_out));
        diagnostics.balance_metric = std::abs(q_in - q_out) / std::max(reference_flux, Real(1.0e-20));
    }

    Real l2_num = Real(0.0);
    Real l2_den = Real(0.0);

    if (cfg.mode == StreamwiseMode::Velocity) {
        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int y = 1; y < cfg.ny - 1; ++y) {
                const int cell = lbm::flatten_xyz(0, y, z, cfg.nx, cfg.ny, cfg.nz);
                const Real target = lbm::prescribed_inlet_velocity_x(y, z, cfg);
                const Real diff = ux[cell] - target;
                l2_num += diff * diff;
                l2_den += target * target;
            }
        }
    } else {
        const Real viscosity = (cfg.tau - Real(0.5)) / Real(3.0);
        const Real forcing = (cfg.mode == StreamwiseMode::PeriodicBodyForce)
            ? cfg.body_force_x
            : lbm::kCs2 * (cfg.rho_inlet - cfg.rho_outlet) / (cfg.rho0 * Real(cfg.nx - 1));

        for (int z = 1; z < cfg.nz - 1; ++z) {
            for (int y = 1; y < cfg.ny - 1; ++y) {
                const int cell = lbm::flatten_xyz(interior_x, y, z, cfg.nx, cfg.ny, cfg.nz);
                const Real exact = rectangular_duct_velocity(forcing, viscosity, y, z, cfg);
                const Real diff = ux[cell] - exact;
                l2_num += diff * diff;
                l2_den += exact * exact;
            }
        }
    }

    diagnostics.l2_error = std::sqrt(l2_num / std::max(l2_den, Real(1.0e-20)));
    return diagnostics;
}

void print_diagnostics(const Diagnostics& diagnostics, StreamwiseMode mode) {
    std::cout << "step=" << diagnostics.step
              << " mode=" << mode_name(mode)
              << " mass=" << std::setprecision(10) << diagnostics.total_mass
              << " mean_rho=" << diagnostics.mean_density
              << " bulk_u=" << diagnostics.bulk_velocity
              << " flow_rate=" << diagnostics.flow_rate
              << " max_ux=" << diagnostics.max_streamwise_velocity
              << " residual=" << diagnostics.residual
              << " l2=" << diagnostics.l2_error
              << " balance=" << diagnostics.balance_metric
              << " mlups=" << diagnostics.mlups_current
              << " mlups[min/avg/max]=" << diagnostics.mlups_min << '/' << diagnostics.mlups_avg << '/' << diagnostics.mlups_max << '\n';
}

void append_diagnostics_csv(std::ofstream& csv, const Diagnostics& diagnostics) {
    csv << diagnostics.step << ','
        << std::setprecision(16) << diagnostics.total_mass << ','
        << diagnostics.mean_density << ','
        << diagnostics.bulk_velocity << ','
        << diagnostics.flow_rate << ','
        << diagnostics.max_streamwise_velocity << ','
        << diagnostics.residual << ','
        << diagnostics.l2_error << ','
        << diagnostics.balance_metric << ','
        << diagnostics.mlups_current << ','
        << diagnostics.mlups_min << ','
        << diagnostics.mlups_avg << ','
        << diagnostics.mlups_max << '\n';
}

void arm_mlups_timer(MlupsStats* mlups, int step) {
    mlups->last_sample_time = MlupsStats::Clock::now();
    mlups->last_step = step;
    mlups->armed = true;
}

void update_mlups_stats(MlupsStats* mlups, int step, int cell_count, Diagnostics* diagnostics) {
    if (!mlups->armed) {
        diagnostics->mlups_current = Real(0.0);
        diagnostics->mlups_min = Real(0.0);
        diagnostics->mlups_avg = Real(0.0);
        diagnostics->mlups_max = Real(0.0);
        return;
    }

    const auto now = MlupsStats::Clock::now();
    const double elapsed_seconds = std::chrono::duration<double>(now - mlups->last_sample_time).count();
    const int step_delta = step - mlups->last_step;
    if (step_delta <= 0 || elapsed_seconds <= 0.0) {
        diagnostics->mlups_current = Real(0.0);
        diagnostics->mlups_min = std::isfinite(mlups->min_mlups) ? static_cast<Real>(mlups->min_mlups) : Real(0.0);
        diagnostics->mlups_avg = (mlups->total_seconds > 0.0) ? static_cast<Real>(mlups->total_updates / mlups->total_seconds / 1.0e6) : Real(0.0);
        diagnostics->mlups_max = static_cast<Real>(mlups->max_mlups);
        return;
    }

    const double updates = static_cast<double>(cell_count) * static_cast<double>(step_delta);
    const double current_mlups = updates / elapsed_seconds / 1.0e6;

    mlups->current_mlups = current_mlups;
    mlups->min_mlups = std::min(mlups->min_mlups, current_mlups);
    mlups->max_mlups = std::max(mlups->max_mlups, current_mlups);
    mlups->total_updates += updates;
    mlups->total_seconds += elapsed_seconds;
    mlups->last_sample_time = now;
    mlups->last_step = step;

    diagnostics->mlups_current = static_cast<Real>(current_mlups);
    diagnostics->mlups_min = static_cast<Real>(mlups->min_mlups);
    diagnostics->mlups_avg = static_cast<Real>(mlups->total_updates / mlups->total_seconds / 1.0e6);
    diagnostics->mlups_max = static_cast<Real>(mlups->max_mlups);
}

void update_shift_vectors(
    const SimulationConfig& cfg,
    std::array<int, lbm::kQ>* sx,
    std::array<int, lbm::kQ>* sy,
    std::array<int, lbm::kQ>* sz) {
    // After collide-and-stream has written into the next logical shift layout,
    // the host advances the cumulative offsets so all subsequent accesses use
    // that new streamed state.
    for (int q = 0; q < lbm::kQ; ++q) {
        (*sx)[q] = lbm::advance_shift_index((*sx)[q], lbm::kCx[q], cfg.nx);
        (*sy)[q] = lbm::advance_shift_index((*sy)[q], lbm::kCy[q], cfg.ny);
        (*sz)[q] = lbm::advance_shift_index((*sz)[q], lbm::kCz[q], cfg.nz);
    }
}

MemoryReport build_memory_report(int cell_count) {
    MemoryReport report;

    const std::size_t q_field_bytes = sizeof(Real) * static_cast<std::size_t>(lbm::kQ) * static_cast<std::size_t>(cell_count);
    const std::size_t scalar_field_bytes = sizeof(Real) * static_cast<std::size_t>(cell_count);
    const std::size_t node_type_bytes = sizeof(std::uint8_t) * static_cast<std::size_t>(cell_count);
    const std::size_t shift_bytes = sizeof(int) * static_cast<std::size_t>(lbm::kQ);

    // Host-side persistent buffers: node map, four macroscopic output fields,
    // and the three logical shift vectors stored on the host.
    report.host_bytes =
        node_type_bytes +
        4 * scalar_field_bytes +
        3 * shift_bytes;

    // GPU-side persistent buffers: one DF storage set, four macro fields used
    // for diagnostics/output, the node-type field, and the shift vectors.
    report.gpu_bytes =
        q_field_bytes +
        4 * scalar_field_bytes +
        node_type_bytes +
        3 * shift_bytes;

    report.total_bytes = report.host_bytes + report.gpu_bytes;
    report.host_bytes_per_cell = static_cast<double>(report.host_bytes) / static_cast<double>(cell_count);
    report.gpu_bytes_per_cell = static_cast<double>(report.gpu_bytes) / static_cast<double>(cell_count);
    report.total_bytes_per_cell = static_cast<double>(report.total_bytes) / static_cast<double>(cell_count);
    return report;
}

void print_memory_report(const MemoryReport& report) {
    std::cout << "Memory demand per simulation"
              << " host=" << format_bytes(report.host_bytes)
              << " gpu=" << format_bytes(report.gpu_bytes)
              << " total=" << format_bytes(report.total_bytes) << '\n';

    std::ostringstream per_cell;
    per_cell << std::fixed << std::setprecision(3)
             << "Memory demand per cell"
             << " host=" << report.host_bytes_per_cell << " B"
             << " gpu=" << report.gpu_bytes_per_cell << " B"
             << " total=" << report.total_bytes_per_cell << " B";
    std::cout << per_cell.str() << '\n';
}

void apply_user_node_type_primitives(std::vector<std::uint8_t>* node_type, const SimulationConfig& cfg) {
    (void)node_type;
    (void)cfg;

    // Host-side geometry stamping hook. Leave it empty for the baseline duct,
    // or add calls such as:
    //
    // lbm::fill_box(node_type, cfg, lbm::kWall, 8, 12, 6, 10, 6, 10);
    // lbm::fill_ball(node_type, cfg, lbm::kWall, 20.0, 12.0, 12.0, 4.0);
    // lbm::fill_plane(node_type, cfg, lbm::kInlet, lbm::PrimitiveAxis::X, 4, 0, 1, cfg.ny - 2, 1, cfg.nz - 2);
    // lbm::fill_cylinder(node_type, cfg, lbm::kWall, lbm::PrimitiveAxis::Z, 64.0, 32.0, 16.0, 0, 63);
    //
    // The node map is uploaded once after this hook and then used by both the
    // equilibrium initialization and all subsequent boundary kernels.
}

}  // namespace

int main(int argc, char** argv) {
    try {
        SimulationConfig cfg;
        RuntimeOptions runtime;
        parse_arguments(argc, argv, &cfg, &runtime);
        validate_config(&cfg);

        std::filesystem::create_directories(runtime.output_dir);

        lbm::copy_lattice_constants_to_device();

        const int cell_count = cfg.nx * cfg.ny * cfg.nz;
        const std::size_t field_bytes = sizeof(Real) * static_cast<std::size_t>(cell_count);
        const MemoryReport memory_report = build_memory_report(cell_count);

        Real* d_f = nullptr;
        Real* d_rho = nullptr;
        Real* d_ux = nullptr;
        Real* d_uy = nullptr;
        Real* d_uz = nullptr;
        std::uint8_t* d_node_type = nullptr;
        int* d_sx = nullptr;
        int* d_sy = nullptr;
        int* d_sz = nullptr;

        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_f), sizeof(Real) * static_cast<std::size_t>(lbm::kQ) * cell_count), "allocate DF field");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rho), field_bytes), "allocate rho field");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ux), field_bytes), "allocate ux field");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_uy), field_bytes), "allocate uy field");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_uz), field_bytes), "allocate uz field");
        lbm::cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&d_node_type), sizeof(std::uint8_t) * static_cast<std::size_t>(cell_count)),
            "allocate node type field");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sx), sizeof(int) * lbm::kQ), "allocate x shifts");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sy), sizeof(int) * lbm::kQ), "allocate y shifts");
        lbm::cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sz), sizeof(int) * lbm::kQ), "allocate z shifts");

        std::array<int, lbm::kQ> shift_x{};
        std::array<int, lbm::kQ> shift_y{};
        std::array<int, lbm::kQ> shift_z{};
        std::vector<std::uint8_t> host_node_type(cell_count, 0);
        std::vector<Real> host_rho(cell_count, Real(0.0));
        std::vector<Real> host_ux(cell_count, Real(0.0));
        std::vector<Real> host_uy(cell_count, Real(0.0));
        std::vector<Real> host_uz(cell_count, Real(0.0));
        lbm::cuda_check(cudaMemcpy(d_sx, shift_x.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "copy initial x shifts");
        lbm::cuda_check(cudaMemcpy(d_sy, shift_y.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "copy initial y shifts");
        lbm::cuda_check(cudaMemcpy(d_sz, shift_z.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "copy initial z shifts");

        lbm::build_default_node_type_map(&host_node_type, cfg);
        apply_user_node_type_primitives(&host_node_type, cfg);
        lbm::cuda_check(
            cudaMemcpy(d_node_type, host_node_type.data(), sizeof(std::uint8_t) * static_cast<std::size_t>(cell_count), cudaMemcpyHostToDevice),
            "copy node type field");
        lbm::launch_initialize_equilibrium(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
        lbm::cuda_check(cudaDeviceSynchronize(), "initialize solver state");
        if (runtime.write_node_map) {
            lbm::write_node_map_vti(runtime.output_dir + "/map.vti", cfg, host_node_type);
        }

        std::ofstream diagnostics_csv(runtime.diagnostics_csv);
        if (!diagnostics_csv) {
            throw std::runtime_error("Unable to open diagnostics CSV: " + runtime.diagnostics_csv);
        }
        diagnostics_csv
            << "step,total_mass,mean_density,bulk_velocity,flow_rate,max_streamwise_velocity,residual,l2_error,balance_metric,mlups_current,mlups_min,mlups_avg,mlups_max\n";

        MlupsStats mlups_stats{};

        auto recover_to_host = [&](int step, bool write_output, Real initial_mass, Real* previous_bulk_velocity) {
            lbm::launch_recover_macros(d_f, d_node_type, cfg, d_sx, d_sy, d_sz, d_rho, d_ux, d_uy, d_uz);
            lbm::cuda_check(cudaDeviceSynchronize(), "recover macroscopic fields");
            lbm::cuda_check(cudaMemcpy(host_rho.data(), d_rho, field_bytes, cudaMemcpyDeviceToHost), "copy rho to host");
            lbm::cuda_check(cudaMemcpy(host_ux.data(), d_ux, field_bytes, cudaMemcpyDeviceToHost), "copy ux to host");
            lbm::cuda_check(cudaMemcpy(host_uy.data(), d_uy, field_bytes, cudaMemcpyDeviceToHost), "copy uy to host");
            lbm::cuda_check(cudaMemcpy(host_uz.data(), d_uz, field_bytes, cudaMemcpyDeviceToHost), "copy uz to host");

            Diagnostics diagnostics = compute_diagnostics(cfg, step, host_node_type, host_rho, host_ux, initial_mass, *previous_bulk_velocity);
            update_mlups_stats(&mlups_stats, step, cell_count, &diagnostics);
            print_diagnostics(diagnostics, cfg.mode);
            append_diagnostics_csv(diagnostics_csv, diagnostics);
            *previous_bulk_velocity = diagnostics.bulk_velocity;

            if (write_output) {
                std::ostringstream stem;
                stem << runtime.output_dir << "/duct_step_" << std::setw(7) << std::setfill('0') << step;
                if (!runtime.do_not_write_full_volume) {
                    lbm::write_vti(stem.str() + ".vti", cfg, host_rho, host_ux, host_uy, host_uz);
                }
                if (runtime.write_cross_sections) {
                    lbm::write_vti_midplane_cross_sections(stem.str(), cfg, host_rho, host_ux, host_uy, host_uz);
                }
            }

            return diagnostics;
        };

        std::cout << "Running D3Q27 single-grid periodic-shift solver"
                  << " nx=" << cfg.nx
                  << " ny=" << cfg.ny
                  << " nz=" << cfg.nz
                  << " steps=" << cfg.nsteps
                  << " mode=" << mode_name(cfg.mode)
                  << " outlet=" << outlet_name(cfg.outlet_mode)
                  << " inlet-profile=" << profile_name(cfg.inlet_profile)
                  << " cross-sections=" << (runtime.write_cross_sections ? "on" : "off")
                  << " node-map=" << (runtime.write_node_map ? "on" : "off")
                  << " do-not-write-full-volume=" << (runtime.do_not_write_full_volume ? "on" : "off")
                  << '\n';
        print_memory_report(memory_report);

        Real previous_bulk_velocity = Real(0.0);
        const Diagnostics initial_diagnostics = recover_to_host(0, true, Real(0.0), &previous_bulk_velocity);
        const Real initial_mass = initial_diagnostics.total_mass;
        arm_mlups_timer(&mlups_stats, 0);
        Diagnostics latest_diagnostics = initial_diagnostics;

        for (int step = 1; step <= cfg.nsteps; ++step) {
            lbm::launch_collide_and_stream(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
            update_shift_vectors(cfg, &shift_x, &shift_y, &shift_z);
            lbm::cuda_check(cudaMemcpy(d_sx, shift_x.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "update x shifts");
            lbm::cuda_check(cudaMemcpy(d_sy, shift_y.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "update y shifts");
            lbm::cuda_check(cudaMemcpy(d_sz, shift_z.data(), sizeof(int) * lbm::kQ, cudaMemcpyHostToDevice), "update z shifts");
            // Boundary reconstruction always runs after the streamed field exists
            // in logical form at the current shift state.
            lbm::launch_apply_wall_boundaries(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
            if (cfg.mode == StreamwiseMode::Pressure) {
                lbm::launch_apply_pressure_boundaries(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
            } else if (cfg.mode == StreamwiseMode::Velocity) {
                lbm::launch_apply_velocity_boundaries(d_f, d_node_type, cfg, d_sx, d_sy, d_sz);
            }

            const bool write_output = (cfg.output_every > 0) && (step % cfg.output_every == 0 || step == cfg.nsteps);
            const bool print_diagnostics = (cfg.diagnostic_every > 0) && (step % cfg.diagnostic_every == 0 || step == cfg.nsteps);
            if (write_output || print_diagnostics) {
                lbm::cuda_check(cudaDeviceSynchronize(), "synchronize timestep before diagnostics");
                latest_diagnostics = recover_to_host(step, write_output, initial_mass, &previous_bulk_velocity);
            }
        }

        std::cout << "MLUPS summary"
                  << " min=" << latest_diagnostics.mlups_min
                  << " avg=" << latest_diagnostics.mlups_avg
                  << " max=" << latest_diagnostics.mlups_max << '\n';

        lbm::cuda_check(cudaFree(d_f), "free DF field");
        lbm::cuda_check(cudaFree(d_rho), "free rho field");
        lbm::cuda_check(cudaFree(d_ux), "free ux field");
        lbm::cuda_check(cudaFree(d_uy), "free uy field");
        lbm::cuda_check(cudaFree(d_uz), "free uz field");
        lbm::cuda_check(cudaFree(d_node_type), "free node type field");
        lbm::cuda_check(cudaFree(d_sx), "free x shifts");
        lbm::cuda_check(cudaFree(d_sy), "free y shifts");
        lbm::cuda_check(cudaFree(d_sz), "free z shifts");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
