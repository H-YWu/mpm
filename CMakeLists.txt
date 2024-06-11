cmake_minimum_required(VERSION 3.27)
project(chains_mpm LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

# Configure third-party libraries through FetchContent
include(FetchContent)
## GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw
)
FetchContent_MakeAvailable(glfw)
## GLAD (OpenGL)
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
)
FetchContent_MakeAvailable(glad)

# -> build libraries 
add_subdirectory(src)