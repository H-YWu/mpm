#include <glad/glad.h>

#include "mpm_solver3d_builder.h"
#include "mpm_solver3d.h"
#include "online_mpm3d_renderer.h"
#include "create_file.h"

#include <GLFW/glfw3.h>
#include <Eigen/Dense>
#include <thrust/device_new.h>
#include <vector>
#include <iostream>
#include <cstdio>

void errorCallback(int error, const char* description) {
    std::cerr << "[ERROR] " << description << std::endl;
}

void processInput(GLFWwindow *window) {
    // Close Windows
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    // 
}

int main(int argc, const char *argv[]) {

    if (argc < 2) {
        printf("Usage: %s path/to/configuration/file\n", argv[0]);
        return 1;
    }
    std::string config_file_path = argv[1]; 
    chains::MPM3DConfiguration config = chains::parseYAML(config_file_path);

    std::string program_name = argv[0];
    int particles_num = -1;
    chains::MPMSolver3D solver = chains::buildMPMSolver3DFromYAML(config, particles_num);

    std::string output_dir;
    if (config.offline) {   // Offline rendering: write to disk
        output_dir = chains::createDirectoryWithPrefixAndCurrentTime(program_name);
    }

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    const unsigned int window_width = 800;
    const unsigned int window_height = 600;

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "MPM PIC-FLIP", nullptr, nullptr);
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

    chains::OnlineMPM3DRenderer renderer(window_width, window_height, particles_num, config.particle_frag_shader_path);

    solver.registerGLBufferWithCUDA(renderer.getMaterilPointVBO());

    // Loop of simulation and rendering 
    int step = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS)
            renderer.viewFromOriginCamera();
        if (glfwGetKey(window, GLFW_KEY_U) == GLFW_PRESS)
            renderer.viewFromUpCamera();
        if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS)
            renderer.viewFromFrontCamera();
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
            renderer.viewFromSideCamera();
        if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
            solver.switchIfEnableParticlesCollision();

        std::cout << "[MPM] step: #" << step << std::endl;

        // Simulation
        solver.simulateOneStep();

        // Online rendering
        solver.updateGLBufferWithCUDA();
        renderer.render();
        
        // Offline rendering
        if (config.offline) {
            std::string file_path = chains::createFileWithPaddingNumberNameInDirectory(step, 5, output_dir);
            solver.writeToFile(file_path);
        }

        // glfw: swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();

        step ++;
    }

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
