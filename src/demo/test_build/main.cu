#include "mpm_solver3d.h"

#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

const unsigned int WIDTH = 800;
const unsigned int HEIGHT = 600;

void errorCallback(int error, const char* description) {
    std::cerr << "Errors: " << description << std::endl;
}

void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main(int argc, const char *argv[]) {

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Test Build", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);

    // glfw setting callback
    glfwSetErrorCallback(errorCallback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return EXIT_FAILURE;
    }

    std::cerr << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    glEnable(GL_DEPTH_TEST);

    chains::MPMSolver3D test_solver;

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        test_solver.step();
        // glfw: swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
