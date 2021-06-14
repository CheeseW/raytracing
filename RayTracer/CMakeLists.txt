# CMakeList.txt : CMake project for RayTracer, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project (RayTracer VERSION 0.1
                   DESCRIPTION "Ray tracer tutorial"
                   LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(gui)
add_subdirectory(img)
add_subdirectory(img_cuda)