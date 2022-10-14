cmake_minimum_required(VERSION 3.10)
project(Supercomputers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

set(GCC_COVERAGE_COMPILE_FLAGS "-Ofast -march=native")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(
    work1 
    hw_1/main.cpp
    vectors_and_matrices/array_types.hpp
)
