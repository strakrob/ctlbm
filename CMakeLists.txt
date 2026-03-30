cmake_minimum_required(VERSION 3.24)

project(periodic_shift_duct_lbm LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75-real CACHE STRING "CUDA architectures to build" FORCE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(LBM_USE_FLOAT "Build all solver variants in single precision." OFF)

find_package(CUDAToolkit REQUIRED)

function(add_lbm_cuda_target target_name include_dir)
  add_executable(${target_name} ${ARGN})
  target_include_directories(${target_name} PRIVATE ${include_dir})
  target_compile_definitions(${target_name} PRIVATE
    $<$<BOOL:${LBM_USE_FLOAT}>:LBM_USE_FLOAT=1>
  )
endfunction()

add_lbm_cuda_target(
  lbmd3q27_cpp
  src
  src/main.cu
  src/lbm_kernels.cu
  src/boundary_conditions.cu
  src/vti_writer.cu
)

target_link_libraries(lbmd3q27_cpp PRIVATE
  CUDA::cuda_driver
)

add_lbm_cuda_target(
  lbmd3q27_c
  src_C
  src_C/main.cu
  src_C/lbm_kernels.cu
)

add_lbm_cuda_target(
  lbmd3q27_c_vmm
  simpleCVMM
  simpleCVMM/main.cu
  simpleCVMM/lbm_kernels.cu
)

target_link_libraries(lbmd3q27_c_vmm PRIVATE
  CUDA::cuda_driver
)
