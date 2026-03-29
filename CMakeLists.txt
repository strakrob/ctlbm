cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build")

project(periodic_shift_duct_lbm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
option(LBM_USE_FLOAT "Build the solver in single precision instead of double precision." OFF)
option(LBM_BENCH_STORE_INPUT "Benchmark collide kernel by storing the pre-collision populations instead of post-collision values." OFF)
option(LBM_BENCH_COMPUTE_ONLY "Benchmark collide kernel by computing post-collision values but suppressing the 27 fluid-node DF stores." OFF)

if(LBM_BENCH_STORE_INPUT AND LBM_BENCH_COMPUTE_ONLY)
  message(FATAL_ERROR "LBM_BENCH_STORE_INPUT and LBM_BENCH_COMPUTE_ONLY are mutually exclusive.")
endif()

add_executable(lbmd3q27
  src/main.cu
  src/lbm_kernels.cu
  src/boundary_conditions.cu
  src/vti_writer.cu
)

target_include_directories(lbmd3q27 PRIVATE src)

target_compile_features(lbmd3q27 PRIVATE cxx_std_17 cuda_std_17)

target_compile_definitions(lbmd3q27 PRIVATE
  $<$<BOOL:${LBM_USE_FLOAT}>:LBM_USE_FLOAT=1>
  $<$<BOOL:${LBM_BENCH_STORE_INPUT}>:LBM_BENCH_STORE_INPUT=1>
  $<$<BOOL:${LBM_BENCH_COMPUTE_ONLY}>:LBM_BENCH_COMPUTE_ONLY=1>
)

target_link_libraries(lbmd3q27 PRIVATE
  CUDA::cuda_driver
)

set_target_properties(lbmd3q27 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
