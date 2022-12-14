if(USE_CUDA)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust)
    file(GLOB_RECURSE PARTI_SRC RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "!algorithms" "*.cpp" "*cu" "algorithms/ttm_cuda.cu" "*.hpp" "include/*.hpp")
    set_source_files_properties(${PARTI_SRC} PROPERTIES LANGUAGE CUDA)
    if(BUILD_SHARED)
        add_library(ParTI SHARED ${PARTI_SRC})
        target_include_directories(ParTI PUBLIC "${CUDA_INCLUDE_DIRS}")
        target_link_libraries(ParTI -lcublas -lcusolver Thrust)
    endif()
    if(BUILD_STATIC)
        add_library(ParTI_s STATIC ${PARTI_SRC})
        target_include_directories(ParTI_s PUBLIC "${CUDA_INCLUDE_DIRS}")
        target_link_libraries(ParTI_s -lcublas -lcusolver Thrust)
    endif()
endif()

if(BUILD_SHARED)
    set_target_properties(ParTI PROPERTIES C_STANDARD 99)
    set_target_properties(ParTI PROPERTIES CXX_STANDARD 11)
endif()
if(BUILD_STATIC)
    set_target_properties(ParTI_s PROPERTIES C_STANDARD 99)
    set_target_properties(ParTI_s PROPERTIES CXX_STANDARD 11)
endif()