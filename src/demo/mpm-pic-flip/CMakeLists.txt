file(GLOB EXE_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cu)
file(GLOB EXE_CPP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

set(EXECUTABLE_NAME mpm-pic-flip)
add_executable(${EXECUTABLE_NAME} ${EXE_CUDA_SRCS} ${EXE_CPP_SRCS})
target_link_libraries(${EXECUTABLE_NAME} PUBLIC ${LIB_CMPM_NAME})
target_include_directories(${EXECUTABLE_NAME} PUBLIC ${INC_DIR})
# Enable relocatable device code mode
set_target_properties(${LIB_CMPM_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)