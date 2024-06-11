set(LIB_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(INC_DIR ${CMAKE_SOURCE_DIR}/include)
set(BIN_DIR ${CMAKE_BINARY_DIR})
# Source code for building libraries
file(GLOB LIB_CPP_SRCS ${LIB_SRC_DIR}/*.cpp)
file(GLOB LIB_CUDA_SRCS ${LIB_SRC_DIR}/*.cu)

# Interface library: compiler flags
## Target: external libraries
add_library(external_libs INTERFACE)
target_link_libraries(external_libs INTERFACE glfw glad)
## Target: general compiler flags
add_library(compiler_flags INTERFACE)
target_compile_features(compiler_flags INTERFACE cxx_std_17)
set(CMAKE_CXX_EXTENSIONS Off)
### Target: compiler flags for developers
add_library(developer_flags INTERFACE)
set(cxx_flags -Wall)
set(cuda_flags -arch=sm_80 -Xcompiler=-Wall -Xcudafe=--display_error_number)
#### Platform-specific & debug/release flags
set(msvc_cxx_debug_flags /MDd /Zi /Ob0 /Od /RTC1)
set(msvc_cxx_release_flags /MD /O2 /Ob2 /DNDEBUG)
set(msvc_cuda_debug_flags --compiler-options /MDd)
set(msvc_cuda_release_flags --compiler-options /MD)
set(gcc_cxx_debug_flags -g -O0)
set(gcc_cxx_release_flags -O3)
set(gcc_cuda_debug_flags --compiler-options -g -O0)
set(gcc_cuda_release_flags --compiler-options -O3)
if (MSVC)
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(cxx_flags "${cxx_flags} ${msvc_cxx_debug_flags}")
        set(cuda_flags "${cuda_flags} ${msvc_cuda_debug_flags}")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        set(cxx_flags "${cxx_flags} ${msvc_cuda_release_flags}")
        set(cuda_flags "${cuda_flags} ${msvc_cuda_release_flags}")
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_BUILD_TYPE MATCHES Debug)
        set(cxx_flags "${cxx_flags} ${gcc_cxx_debug_flags}")
        set(cuda_flags "${cuda_flags} ${gcc_cuda_debug_flags}")
    elseif(CMAKE_BUILD_TYPE MATCHES Release)
        set(cxx_flags "${cxx_flags} ${gcc_cuda_release_flags}")
        set(cuda_flags "${cuda_flags} ${gcc_cuda_release_flags}")
    endif()
endif()
target_compile_options(developer_flags INTERFACE
    # CXX builds
    $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
    # CUDA builds
    $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)
### Add developer flags to compiler flags
target_link_libraries(compiler_flags INTERFACE
    $<BUILD_INTERFACE:developer_flags>
)

# Compile source files into a library
set(LIB_MPM_NAME mpm_lib)
add_library(${LIB_MPM_NAME} STATIC ${LIB_CUDA_SRCS} ${LIB_CPP_SRCS})
target_link_libraries(${LIB_MPM_NAME} PUBLIC compiler_flags external_libs)
target_include_directories(${LIB_MPM_NAME} PUBLIC ${INC_DIR})

# -> build executables
add_subdirectory(demo)