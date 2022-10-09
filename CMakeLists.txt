cmake_minimum_required(VERSION 3.10)
project(Supercomputers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

add_executable(
    work1 
    hw_1/main.cpp
    vectors_and_matrices/array_types.hpp
    vectors_and_matrices/linalg.hpp
)
