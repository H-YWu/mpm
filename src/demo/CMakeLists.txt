# -> find & build each executable
file(GLOB DEMO_SUBDIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(DEMO_SUBDIR ${DEMO_SUBDIRS})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DEMO_SUBDIR})
        add_subdirectory(${DEMO_SUBDIR})
    endif()
endforeach()