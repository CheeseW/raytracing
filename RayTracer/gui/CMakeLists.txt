set(EXE_NAME app)
add_executable(${EXE_NAME} app.cpp)

target_include_directories(${EXE_NAME}
	PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

# target_compile_options(${EXE_NAME} PRIVATE -Werror)
target_compile_features(${EXE_NAME} PRIVATE cxx_std_11)

find_package(GLEW REQUIRED)
find_package(GLFW REQUIRED)
find_package(OpenGL REQUIRED)


target_link_libraries(${EXE_NAME} GLFW::GLFW GLEW::GLEW OpenGL::GL)