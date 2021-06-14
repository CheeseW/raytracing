set(EXE_NAME cuda_test)
add_executable(${EXE_NAME} main.cu "ray.h" "vec3.h")

target_include_directories(${EXE_NAME}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(${EXE_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${EXE_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_features(${EXE_NAME} PRIVATE cxx_std_11)
set_property(TARGET ${EXE_NAME} PROPERTY CUDA_STANDARD 11)

