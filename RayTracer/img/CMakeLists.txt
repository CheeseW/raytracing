set(EXE_NAME img_test)
add_executable(${EXE_NAME} main.cpp "color.h" "ray.h" "hittable.h" "sphere.h" "hittable_list.h" "utilities.h" "camera.h")

target_include_directories(${EXE_NAME}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(${EXE_NAME} PRIVATE cxx_std_11)
find_package(Eigen3 REQUIRED)

target_link_libraries(${EXE_NAME} Eigen3::Eigen)