cmake_minimum_required(VERSION 3.27)
project(chains_mpm LANGUAGES CXX CUDA)

# CUDA
include(CheckLanguage)
check_language(CUDA)

find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION_STRING}")
else()
    message(FATAL_ERROR "CUDA Toolkit not found. Please check your CUDA installation.")
endif()

# OpenVDB
if(WIN32)
    list(APPEND CMAKE_MODULE_PATH "C:/Program Files/OpenVDB/lib/cmake/OpenVDB")
elseif(UNIX)
    list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
else()
    message(FATAL_ERROR "Unsupported operating system")
endif()
## Find OpenVDB package
find_package(OpenVDB REQUIRED)
if(OpenVDB_FOUND)
    message(STATUS "Found OpenVDB: ${OpenVDB_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenVDB not found. Please check your installation.")
endif()

# Configure third-party libraries through FetchContent
include(FetchContent)
## GLFW
FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(glfw)
## GLAD (OpenGL)
FetchContent_Declare(
	glad
	GIT_REPOSITORY https://github.com/Dav1dde/glad.git
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(glad)
## Eigen
FetchContent_Declare(
	eigen
	GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
	GIT_TAG 3.4
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(eigen)
## yaml-cpp
FetchContent_Declare(
	yaml-cpp
	GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
	GIT_TAG yaml-cpp-0.7.0
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
set(YAML_CPP_BUILD_TESTS OFF)
FetchContent_MakeAvailable(yaml-cpp)
## glm
FetchContent_Declare(
	glm
	GIT_REPOSITORY	https://github.com/g-truc/glm.git
	GIT_TAG 	bf71a834948186f4097caa076cd2663c69a10e1e #refs/tags/1.0.1
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(glm)
## ImGui
FetchContent_Declare(
	imgui
	GIT_REPOSITORY https://github.com/ocornut/imgui.git
	GIT_TAG v1.90.8
	GIT_SHALLOW TRUE
	GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(imgui)

# -> build libraries 
add_subdirectory(src)