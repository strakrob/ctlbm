cmake_minimum_required(VERSION 3.20)
project(lbmd3q27_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

add_executable(lbmd3q27 src/main.cu)
set_target_properties(lbmd3q27 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(lbmd3q27 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
