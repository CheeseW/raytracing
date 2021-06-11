#define GLFW_INLCUDE_NONE
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

static void error_callback(int error, const char* description) {
	std::cerr << "Error:" << description << std::endl;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GLFW_TRUE);
}

int main(int argc, char* argv[]) {
	if (!glfwInit()) {
		std::cerr << "Cannot initialize glfw" << std::endl;
		exit(EXIT_FAILURE);
	}

	glfwSetErrorCallback(error_callback);


	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

	constexpr int w = 640;
	constexpr int h = 480;

	GLFWwindow* window = glfwCreateWindow(w, h, "My Title", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		std::cerr << "Window or context creation failed" << std::endl;
		exit(EXIT_FAILURE);
	}

	glfwSetKeyCallback(window, key_callback);
	glfwMakeContextCurrent(window);

	// load gl with glew
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "Failed to initialize GLEW" << std::endl;
		exit(EXIT_FAILURE);
	}

	glfwSwapInterval(1);

	//glGenBuffers(1, &vertex_buffer)


	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

	double time = glfwGetTime();



	while (!glfwWindowShouldClose(window)) {
		//glfwSwapBuffers();
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	return EXIT_SUCCESS;
}