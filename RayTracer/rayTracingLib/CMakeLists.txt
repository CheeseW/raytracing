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