#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lbm {

#ifdef LBM_USE_FLOAT
using Real = float;
inline constexpr const char* kVtiScalarType = "Float32";
#else
using Real = double;
inline constexpr const char* kVtiScalarType = "Float64";
#endif

constexpr int kQ = 27;
constexpr int kBlockSize = 128;
constexpr Real kCs2 = Real(1.0 / 3.0);
constexpr Real kInvCs2 = Real(3.0);
constexpr Real kInvCs4 = Real(9.0);

// The solver keeps one physical distribution storage array of size Q * N.
// Streaming is expressed purely by changing per-direction logical shifts.
// A logical access (q, x, y, z) is translated to a physical cell index with
// `physical_cell_index`, so reviewers can verify that no second DF field exists.

enum class StreamwiseMode : int {
    PeriodicBodyForce = 0,
    Pressure = 1,
    Velocity = 2,
};

enum class OutletMode : int {
    Extrapolation = 0,
    ZeroGaugePressure = 1,
};

enum class InletProfile : int {
    Uniform = 0,
    Parabolic = 1,
};

enum NodeType : std::uint8_t {
    kFluid = 0,
    kWall = 1,
    kInlet = 2,
    kOutlet = 3,
};

struct SimulationConfig {
    int nx = 64;
    int ny = 24;
    int nz = 24;
    int nsteps = 4000;
    int output_every = 1000;
    int diagnostic_every = 200;
    int analytical_terms = 51;
    Real tau = Real(0.8);
    Real omega = Real(1.25);
    Real rho0 = Real(1.0);
    Real body_force_x = Real(1.0e-6);
    Real rho_inlet = Real(1.001);
    Real rho_outlet = Real(0.999);
    Real inlet_velocity = Real(0.02);
    StreamwiseMode mode = StreamwiseMode::PeriodicBodyForce;
    OutletMode outlet_mode = OutletMode::Extrapolation;
    InletProfile inlet_profile = InletProfile::Parabolic;
};

inline constexpr std::array<int, kQ> kCx = {
    0,
    1, -1, 0, 0, 0, 0,
    1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0,
    1, -1, 1, -1, 1, -1, 1, -1};

inline constexpr std::array<int, kQ> kCy = {
    0,
    0, 0, 1, -1, 0, 0,
    1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1,
    1, -1, 1, -1, -1, 1, -1, 1};

inline constexpr std::array<int, kQ> kCz = {
    0,
    0, 0, 0, 0, 1, -1,
    0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1,
    1, -1, -1, 1, 1, -1, -1, 1};

inline constexpr std::array<int, kQ> kOpp = {
    0,
    2, 1, 4, 3, 6, 5,
    8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17,
    20, 19, 22, 21, 24, 23, 26, 25};

inline constexpr std::array<Real, kQ> kWeights = {
    Real(8.0 / 27.0),
    Real(2.0 / 27.0), Real(2.0 / 27.0), Real(2.0 / 27.0), Real(2.0 / 27.0), Real(2.0 / 27.0), Real(2.0 / 27.0),
    Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0),
    Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0), Real(1.0 / 54.0),
    Real(1.0 / 216.0), Real(1.0 / 216.0), Real(1.0 / 216.0), Real(1.0 / 216.0),
    Real(1.0 / 216.0), Real(1.0 / 216.0), Real(1.0 / 216.0), Real(1.0 / 216.0)};

extern __device__ __constant__ int g_cx[kQ];
extern __device__ __constant__ int g_cy[kQ];
extern __device__ __constant__ int g_cz[kQ];
extern __device__ __constant__ int g_opp[kQ];
extern __device__ __constant__ Real g_w[kQ];

inline void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

__host__ __device__ inline int wrap_index(int value, int extent) {
    value %= extent;
    if (value < 0) {
        value += extent;
    }
    return value;
}

__host__ __device__ inline int flatten_xyz(int x, int y, int z, int nx, int ny, int nz) {
    (void)nz;
    return (z * ny + y) * nx + x;
}

__host__ __device__ inline int distribution_index(int q, int cell, int cell_count) {
    return q * cell_count + cell;
}

__host__ __device__ inline int physical_cell_index(
    int q,
    int x,
    int y,
    int z,
    int nx,
    int ny,
    int nz,
    const int* sx,
    const int* sy,
    const int* sz) {
    const int px = wrap_index(x - sx[q], nx);
    const int py = wrap_index(y - sy[q], ny);
    const int pz = wrap_index(z - sz[q], nz);
    return flatten_xyz(px, py, pz, nx, ny, nz);
}

__host__ __device__ inline int interior_area(const SimulationConfig& cfg) {
    return (cfg.ny - 2) * (cfg.nz - 2);
}

__host__ __device__ inline Real prescribed_inlet_velocity_x(int y, int z, const SimulationConfig& cfg) {
    if (cfg.inlet_profile == InletProfile::Uniform) {
        return cfg.inlet_velocity;
    }

    // The parabolic option is a separable duct profile surrogate.
    // It is zero on the y/z walls and peaks at the duct centre.
    const Real ly = Real(cfg.ny - 1);
    const Real lz = Real(cfg.nz - 1);
    const Real yn = Real(y) / ly;
    const Real zn = Real(z) / lz;
    return Real(16.0) * cfg.inlet_velocity * yn * (Real(1.0) - yn) * zn * (Real(1.0) - zn);
}

void copy_lattice_constants_to_device();

void launch_classify_nodes(std::uint8_t* d_node_type, const SimulationConfig& cfg);
void launch_initialize_equilibrium(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz);
void launch_collide_and_stream(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz);
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
    Real* d_uz);

void launch_apply_wall_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz);
void launch_apply_pressure_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz);
void launch_apply_velocity_boundaries(
    Real* d_f,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    const int* d_sx,
    const int* d_sy,
    const int* d_sz);

void write_vti(
    const std::string& filename,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz);

}  // namespace lbm
