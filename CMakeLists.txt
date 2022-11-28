cmake_minimum_required(VERSION 3.10)
project(Supercomputers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(GCC_COVERAGE_COMPILE_FLAGS "-Ofast -fopenmp -march=native")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(
    work1 
    hw_1/main.cpp
    vectors_and_matrices/array_types.hpp
)

add_executable(
    work2
    hw_2/main.cpp
    vectors_and_matrices/array_types.hpp
    code_templates/1.2-wave_1d.cpp
    code_templates/2.2a-buffon_pi.cpp
    code_templates/2.1b-pi_integ.cpp
)

add_executable(
    2.1b
    code_templates/2.1b-pi_integ.cpp
)

add_executable(
    1.2
    code_templates/1.2-wave_1d.cpp
    vectors_and_matrices/array_types.hpp
)