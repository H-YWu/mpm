set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(EXT_INC_DIR ${CMAKE_SOURCE_DIR}/external)
set(BIN_DIR ${CMAKE_BINARY_DIR})
# Source code for building libraries
file(GLOB LIB_CPP_SRCS ${LIB_SRC_DIR}/*.cpp)
file(GLOB LIB_CUDA_SRCS ${LIB_SRC_DIR}/*.cu)

# Interface library: compiler flags
set(IMGUI_SRC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(imgui_lib ${IMGUI_SRC})
# ImGui headers
target_include_directories(imgui_lib PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends/)
## Target: external libraries
add_library(external_libs INTERFACE)
target_link_libraries(external_libs INTERFACE CUDA::cudart CUDA::cuda_driver cudadevrt glfw glad Eigen3::Eigen yaml-cpp glm::glm imgui_lib)
## Target: general compiler flags
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS Off)
### Target: compiler flags for developers
add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags --extended-lambda --expt-relaxed-constexpr -arch=sm_80 -Xcompiler=-Wall -Xcudafe=--display_error_number -rdc=true)
#### Platform-specific & debug/release flags
set(msvc_cxx_debug_flags /MDd /Zi /Ob0 /Od /RTC1)
set(msvc_cxx_release_flags /MD /O2 /Ob2 /DNDEBUG)
set(msvc_cuda_debug_flags --debug --device-debug --compiler-options /MDd)
set(msvc_cuda_release_flags --compiler-options /MD)
set(gcc_cxx_debug_flags -g -O0)
set(gcc_cxx_release_flags -O3)
set(gcc_cuda_debug_flags --debug --device-debug)
set(gcc_cuda_release_flags --optimize 3 --dopt 3)
#### Apply common flags
target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>)
target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>)
#### Apply platform-sepcific flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${msvc_cxx_debug_flags}>)
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${msvc_cuda_debug_flags}>)
    else()
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${msvc_cxx_release_flags}>)
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${msvc_cuda_release_flags}>)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${gcc_cxx_debug_flags}>)
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${gcc_cuda_debug_flags}>)
    else()
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${gcc_cxx_release_flags}>)
        target_compile_options(developer_flags INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${gcc_cuda_release_flags}>)
    endif()
endif()
### Add developer flags to compiler flags
target_link_libraries(compiler_flags INTERFACE
    $<BUILD_INTERFACE:developer_flags>
)

# Compile source files into a library
set(LIB_CMPM_NAME cmpm)
add_library(${LIB_CMPM_NAME} STATIC ${LIB_CUDA_SRCS} ${LIB_CPP_SRCS})
target_link_libraries(${LIB_CMPM_NAME} PUBLIC compiler_flags external_libs)
target_include_directories(${LIB_CMPM_NAME} PUBLIC ${INC_DIR} ${EXT_INC_DIR} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
# Enable relocatable device code mode
set_target_properties(${LIB_CMPM_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# -> build executables
add_subdirectory(demo)