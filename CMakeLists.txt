cmake_minimum_required(VERSION 3.18)  # CMake 3.2 supports CUDA 7.0
project(ParTI LANGUAGES CUDA CXX)

option(BUILD_STATIC "Build static library" ON)
option(BUILD_SHARED "Build shared library" OFF)

option(USE_CUDA "Use NVIDIA CUDA library" ON)
option(USE_OPENBLAS "Use OpenBLAS library" OFF)
option(USE_LAPACKE "Use LAPACKE library" OFF)
option(USE_MAGMA "Use MAGMA library" OFF)
option(USE_MKL "Use Intel MKL library" ON)

option(CUDA_HOST_COMPILER "CUDA host compiler (gcc < 6.0)" /usr/bin/gcc-5)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
            STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
            "Debug" "Release" "RelWithDebugInfo")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Backward_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/backward-cpp")
add_subdirectory("${Backward_DIR}")

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES "70;80")
    add_definitions(-DPARTI_USE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
    if(CUDA_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3")
    else()
        set(CMAKE_CUDA_FLAS "${CMAKE_CUDA_FLAGS} -g -G -Og")
    endif()
    set(CMAKE_BUILD_RPATH "/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/lib64/;${CMAKE_BUILD_RPATH}")
endif()

# A bug in FindCUDA module forces us to put "link_libraries" before "cuda_add_library".
# So we cannot use "target_link_libraries" for target-wise library tracking.
if(USE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    add_definitions(-DPARTI_USE_OPENBLAS)
    add_definitions(-DPARTI_USE_CBLAS)
    add_definitions(-DPARTI_USE_LAPACKE)
    include_directories(${OpenBLAS_INCLUDE_DIRS})
    link_libraries(${OpenBLAS_LIBRARIES})
endif()
if(USE_CBLAS AND NOT USE_OPENBLAS)
    find_library(
        CBLAS_LIBRARIES
        NAMES cblas
        PATHS ${LAPACKE_DIR}
    )
    add_definitions(-DPARTI_USE_CBLAS)
    get_filename_component(CBLAS_LIBDIR "${CBLAS_LIBRARIES}" DIRECTORY)
    include_directories("${CBLAS_LIBDIR}/../include")
    link_libraries(${CBLAS_LIBRARIES})
endif()
if(USE_LAPACKE AND NOT USE_OPENBLAS)
    find_library(
        LAPACKE_LIBRARIES
        NAMES lapacke
        PATHS ${LAPACKE_DIR}
    )
    add_definitions(-DPARTI_USE_LAPACKE)
    get_filename_component(LAPACKE_LIBDIR "${LAPACKE_LIBRARIES}" DIRECTORY)
    include_directories("${LAPACKE_LIBDIR}/../include")
    link_libraries(${LAPACKE_LIBRARIES})
endif()
if(USE_MKL)
    find_package(MKL REQUIRED)
    add_definitions(-DPARTI_USE_MKL)
    include_directories(${MKL_INCLUDE_DIRS})
    set(MKL_LIBRARIES "mkl_rt")
    link_libraries(${MKL_LIBRARIES})
endif()

if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -fopenmp")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /EHsc /W2")
endif()
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fopenmp")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /W2")
endif()

# A bug in FindCUDA module forces us to put "include_directories" before "cuda_add_library".
# So we cannot use "target_include_directories" for target-wise include tracking.
include_directories("include")
if(NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    link_libraries("m")
endif()

add_subdirectory("src")
add_subdirectory("tests")
