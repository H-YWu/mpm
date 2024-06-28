#include <glad/glad.h>

#include "mpm_solver3d_builder.h"
#include "apic_mpm_solver3d.h"
#include "online_mpm3d_renderer.h"
#include "create_file.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#define GL_SILENCE_DEPRECATION
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include <Eigen/Dense>
#include <thrust/device_new.h>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>

bool isRunningOnWSL() {
    const char* wslEnv = std::getenv("WSLENV");
    return (wslEnv != nullptr);
}

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

    int particles_num = -1;
    chains::APICMPMSolver3D solver = chains::buildAPICMPMSolver3DFromYAML(config, particles_num);

    std::string output_dir;
    if (config.offline) {   // Offline rendering: write to disk
        std::filesystem::path program_path = argv[0];
        std::filesystem::path file_path = argv[1];
        std::string output_dir_name = program_path.stem().string() + "-" + file_path.stem().string();
        output_dir = chains::createNewDirectoryInParentDirectoryWithPrefixAndCurrentTime("../output", output_dir_name);
    }

    // glfw: initialize and configure
    if (!glfwInit()) return EXIT_FAILURE;

#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100
    const char* glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#else
    const char* glsl_version = "#version 410";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
#endif

    const unsigned int window_width = 1280;
    const unsigned int window_height = 720;

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(window_width, window_height, "MPM APIC", nullptr, nullptr);
    if (!window) {
        std::cerr << "[ERROR] Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync: 
        // Vertical synchronization (vsync) is a display option that
        //  synchronizes the frame rate of a game or application with the refresh rate of the monitor.

    // glfw setting callback
    glfwSetErrorCallback(errorCallback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "[ERROR] Failed to initialize GLAD" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "[INFO] OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    glEnable(GL_DEPTH_TEST);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsLight();

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
#ifdef __EMSCRIPTEN__
    ImGui_ImplGlfw_InstallEmscriptenCanvasResizeCallback("#canvas");
#endif
    ImGui_ImplOpenGL3_Init(glsl_version);

    bool run_on_WSL = false;
    if (isRunningOnWSL()) {
        std::cout << "[INFO] running on WSL" << std::endl;
        run_on_WSL = true;
    } else {
        std::cout << "[INFO] running on other OS" << std::endl;
        run_on_WSL = false;
    }

    Eigen::Vector3f grid_origin = config.origin;
    float grid_stride = static_cast<float>(config.stride);
    Eigen::Vector3f grid_target = grid_origin + grid_stride*config.resolution.cast<float>();

    chains::OnlineMPM3DRenderer renderer(
        window_width,
        window_height,
        particles_num,
        grid_origin,
        grid_target,
        grid_stride,
        config.particle_frag_shader_path
    );

    if (run_on_WSL) {
        solver.saveGLBuffer(renderer.getMaterilPointVBO());
    } else {
        solver.registerGLBufferWithCUDA(renderer.getMaterilPointVBO());
    }


    // Loop of simulation and rendering 
    int step = 0;
    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        
        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Information Window 
        {
            ImGui::Begin("Information");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::Text("Simulation step #%d", step);
            bool isParticleCollisionEnabled = solver._enable_particles_collision;
            if (ImGui::Checkbox("Material Point Collision", &isParticleCollisionEnabled)) {
                solver.switchIfEnableParticlesCollision();
            }
            ImGui::End();
        }

        // Simulation
        solver.simulateOneStep();

        // Online rendering
        if (run_on_WSL) {
            solver.updateGLBufferByCPU();
        } else {
            solver.updateGLBufferWithCUDA();
        }
        ImGui::Render();

        renderer.render();
 
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
        // Offline rendering
        if (config.offline) {
            std::string file_path = chains::createFileWithPaddingNumberNameInDirectory(step, 5, output_dir, ".vdb");
            solver.writeToOpenVDB(file_path);
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
