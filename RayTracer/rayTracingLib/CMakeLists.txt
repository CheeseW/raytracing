cmake_minimum_required (VERSION 3.8)


project(rayTracingLib
VERSION 0.1 LANGUAGES CXX CUDA)

# library rayTracing
add_library(rayTracing 
"common/rayTracing.cpp" 
"src/rayTracing_scalar.cpp"
 "src/hittable_list.cpp"
"src/ray.h" 
"src/hittable.h" 
"src/sphere.h" 
"src/hittable_list.h" 
"src/utilities.h" 
"src/camera.h" 
"src/materials.h"
)

target_include_directories(rayTracing PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(rayTracing PRIVATE cxx_std_11)
find_package(Eigen3 REQUIRED)

target_link_libraries(rayTracing Eigen3::Eigen)

# export(TARGETS rayTracing FILE RayTracingConfig.cmake)

add_executable(testlib 
"test/testlib.cpp")

target_link_libraries(testlib rayTracing)


#library rayTracing_cuda
add_library(rayTracing_cuda
"common/rayTracing.cpp"
"cuda_src/rayTracing.cu" 
"cuda_src/ray.h" 
"cuda_src/hittable.h" 
"cuda_src/sphere.h" 
"cuda_src/hittable_list.h" 
"cuda_src/utilities.h" 
 "cuda_src/camera.h" 
 "cuda_src/materials.h")

target_include_directories(rayTracing_cuda PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_src
)

set_target_properties(rayTracing_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(rayTracing_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(rayTracing_cuda PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_compile_features(rayTracing_cuda PRIVATE cxx_std_11)
set_property(TARGET rayTracing_cuda PROPERTY CUDA_STANDARD 11)

# export(TARGETS rayTracing_cuda FILE RayTracingConfig.cmake)

add_executable(testlib_cuda 
"test/testlib_cuda.cpp")

target_link_libraries(testlib_cuda rayTracing_cuda)