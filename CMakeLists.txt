cmake_minimum_required(VERSION 3.5)
project(cuda_test)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

CUDA_ADD_EXECUTABLE(test_cuda main.cpp cudafunc.cu)  
