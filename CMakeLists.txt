cmake_minimum_required(VERSION 3.10)
project(Supercomputers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(GCC_COVERAGE_COMPILE_FLAGS "-O3 -ffast-math -fopenmp -march=native")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(
    work1 
    hw_1/main.cpp
    vectors_and_matrices/array_types.hpp
)

# add_executable(
#     2.1b
#     code_templates/2.1b-pi_integ.cpp
# )

add_executable(
    hw_2_1.2
    hw_2/1.2-wave_1d.cpp
    vectors_and_matrices/array_types.hpp
)

add_executable(
    hw_2_2.2a
    hw_2/2.2a-buffon_pi.cpp
    vectors_and_matrices/array_types.hpp
)

add_executable(
    hw_2_2.1b
    hw_2/2.1b-pi_integ.cpp
    vectors_and_matrices/array_types.hpp
)
