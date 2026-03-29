#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
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
#ifdef LBM_USE_FLOAT
constexpr int kBlockSize = 64;
#else
constexpr int kBlockSize = 128;
#endif
constexpr int kLaunchBounds = kBlockSize;
constexpr Real kCs2 = Real(1.0 / 3.0);
constexpr Real kInvCs2 = Real(3.0);
constexpr Real kInvCs4 = Real(9.0);

#if defined(__CUDACC__)
#define LBM_FORCEINLINE __forceinline__
#define LBM_RESTRICT __restrict__
#else
#define LBM_FORCEINLINE inline
#define LBM_RESTRICT
#endif

#define LBM_FLATTEN_XYZ(x, y, z, nx, ny, nz) ((((z) * (ny)) + (y)) * (nx) + (x))
#define LBM_DISTRIBUTION_INDEX(q, cell, cell_count) ((q) * (cell_count) + (cell))

// The solver keeps one physical population array per D3Q27 direction.
// Each population array is mapped twice into a reserved GPU virtual address
// range so streaming can be realized by advancing one per-population start
// pointer. No ping-pong DF storage or source/destination swap exists.

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

enum class PrimitiveAxis : int {
    X = 0,
    Y = 1,
    Z = 2,
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

struct StreamingView {
    Real* p0 = nullptr;
    Real* p1 = nullptr;
    Real* p2 = nullptr;
    Real* p3 = nullptr;
    Real* p4 = nullptr;
    Real* p5 = nullptr;
    Real* p6 = nullptr;
    Real* p7 = nullptr;
    Real* p8 = nullptr;
    Real* p9 = nullptr;
    Real* p10 = nullptr;
    Real* p11 = nullptr;
    Real* p12 = nullptr;
    Real* p13 = nullptr;
    Real* p14 = nullptr;
    Real* p15 = nullptr;
    Real* p16 = nullptr;
    Real* p17 = nullptr;
    Real* p18 = nullptr;
    Real* p19 = nullptr;
    Real* p20 = nullptr;
    Real* p21 = nullptr;
    Real* p22 = nullptr;
    Real* p23 = nullptr;
    Real* p24 = nullptr;
    Real* p25 = nullptr;
    Real* p26 = nullptr;
    const int* offset = nullptr;
    int logical_cells = 0;
};

__host__ __device__ LBM_FORCEINLINE Real* population_pointer(const StreamingView& view, int q) {
    switch (q) {
        case 0: return view.p0;
        case 1: return view.p1;
        case 2: return view.p2;
        case 3: return view.p3;
        case 4: return view.p4;
        case 5: return view.p5;
        case 6: return view.p6;
        case 7: return view.p7;
        case 8: return view.p8;
        case 9: return view.p9;
        case 10: return view.p10;
        case 11: return view.p11;
        case 12: return view.p12;
        case 13: return view.p13;
        case 14: return view.p14;
        case 15: return view.p15;
        case 16: return view.p16;
        case 17: return view.p17;
        case 18: return view.p18;
        case 19: return view.p19;
        case 20: return view.p20;
        case 21: return view.p21;
        case 22: return view.p22;
        case 23: return view.p23;
        case 24: return view.p24;
        case 25: return view.p25;
        case 26: return view.p26;
    }
    return nullptr;
}

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

__host__ __device__ LBM_FORCEINLINE int flatten_xyz(int x, int y, int z, int nx, int ny, int nz) {
    return LBM_FLATTEN_XYZ(x, y, z, nx, ny, nz);
}

__host__ __device__ LBM_FORCEINLINE int distribution_index(int q, int cell, int cell_count) {
    return LBM_DISTRIBUTION_INDEX(q, cell, cell_count);
}

inline int streaming_linear_offset(int q, const SimulationConfig& cfg) {
    return kCx[q] + kCy[q] * cfg.nx + kCz[q] * cfg.nx * cfg.ny;
}

__host__ __device__ LBM_FORCEINLINE int interior_area(const SimulationConfig& cfg) {
    return (cfg.ny - 2) * (cfg.nz - 2);
}

__host__ __device__ LBM_FORCEINLINE Real prescribed_inlet_velocity_x(int y, int z, const SimulationConfig& cfg) {
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
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg);
void launch_collide_and_stream(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg);
void launch_recover_macros(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg,
    Real* d_rho,
    Real* d_ux,
    Real* d_uy,
    Real* d_uz);

void launch_apply_wall_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg);
void launch_apply_periodic_x_boundaries(
    StreamingView view,
    const SimulationConfig& cfg,
    Real* d_xmin_plane,
    Real* d_xmax_plane);
void launch_apply_pressure_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg);
void launch_apply_velocity_boundaries(
    StreamingView view,
    const std::uint8_t* d_node_type,
    const SimulationConfig& cfg);

void build_default_node_type_map(std::vector<std::uint8_t>* node_type, const SimulationConfig& cfg);

// Host-side map editing helpers for stamping interior obstacle / boundary
// primitives into the node-type field before it is uploaded to the GPU.
void fill_box(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    int x0,
    int x1,
    int y0,
    int y1,
    int z0,
    int z1);
void fill_ball(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    Real center_x,
    Real center_y,
    Real center_z,
    Real radius);
void fill_plane(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    PrimitiveAxis normal_axis,
    int coordinate,
    int half_thickness,
    int span0_begin,
    int span0_end,
    int span1_begin,
    int span1_end);
void fill_cylinder(
    std::vector<std::uint8_t>* node_type,
    const SimulationConfig& cfg,
    NodeType fill_type,
    PrimitiveAxis axis,
    Real center_a,
    Real center_b,
    Real radius,
    int axis_begin,
    int axis_end);

void write_vti(
    const std::string& filename,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz);
void write_vti_midplane_cross_sections(
    const std::string& output_stem,
    const SimulationConfig& cfg,
    const std::vector<Real>& rho,
    const std::vector<Real>& ux,
    const std::vector<Real>& uy,
    const std::vector<Real>& uz);
void write_node_map_vti(
    const std::string& filename,
    const SimulationConfig& cfg,
    const std::vector<std::uint8_t>& node_type);

}  // namespace lbm
