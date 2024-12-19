cmake_minimum_required(VERSION 3.20)
project(tensorbench
        VERSION 0.1.0
        DESCRIPTION "Benchmarks for tensor contractions"
        HOMEPAGE_URL "")

include(cmake/Deprecation.cmake) # Warn if user passes deprecated settings

# Make sure we use TB's own find modules
list(INSERT CMAKE_MODULE_PATH 0  ${PROJECT_SOURCE_DIR}/cmake)

# Set options this build
option(BUILD_SHARED_LIBS           "Build shared libraries"                                       OFF)
option(TB_EIGEN3_BLAS              "Enable BLAS backend for Eigen3 (if OpenBLAS/MKL is enabled)"  OFF)
option(TB_ENABLE_EIGEN1            "Benchmark contractions on CPU with Eigen mode 1"              OFF)
option(TB_ENABLE_EIGEN2            "Benchmark contractions on CPU with Eigen mode 2"              OFF)
option(TB_ENABLE_EIGEN3            "Benchmark contractions on CPU with Eigen mode 3"              OFF)
option(TB_ENABLE_CUDA              "Benchmark contractions on GPU"                                OFF)
option(TB_ENABLE_CUTENSOR          "Benchmark contractions on GPU with CUTENSOR"                  OFF)
option(TB_ENABLE_XTENSOR           "Benchmark contractions on CPU with xtensor"                   OFF)
option(TB_ENABLE_TBLIS             "Benchmark contractions on CPU with TBLIS"                     OFF)
option(TB_ENABLE_CYCLOPS           "Benchmark contractions on CPU with CYCLOPS"                   OFF)
option(COMPILER_ENABLE_ASAN        "Enable sanitizer -fsanitize=address"                          OFF)
option(COMPILER_ENABLE_USAN        "Enable sanitizer -fsanitize=undefined"                        OFF)
option(THREADS_PREFER_PTHREAD_FLAG "Prefer -pthread flag over -lpthread or similar"               ON)


################################################################
### Get environment, host, build, exec and git details       ###
### Generates a header env/environment.h                     ###
### Include it using #include <env/environment.h>            ###
################################################################
include(cmake/environment.cmake)


############################################################
### Print extra info during CMake configure              ###
############################################################
include(cmake/PrintHostInfo.cmake)
include(cmake/PrintOptionInfo.cmake)

##############################
###  Apply  compile flags  ###
##############################
include(cmake/CompilerFlags.cmake)



################################
###  Add all source files    ###
################################
add_executable(tensorbench
               source/main.cpp
               source/config/config.cpp
               source/config/config.cu
               source/debug/info.cpp
               source/debug/stacktrace.cpp
               source/tools/prof.cpp
               source/tools/log.cpp
               source/tools/class_tic_toc.cpp
               source/math/tenx/threads.cpp
               source/mpi/mpi-tools.cpp
               source/tid/tid.cpp
               source/tid/token.cpp
               source/tid/ur.cpp
               source/storage/results.cpp
               source/benchmark/run.cpp
               )

target_include_directories(tensorbench PRIVATE source)
target_enable_static_libgcc(tensorbench)

###########################################
### Fetch or find all the dependencies  ###
###########################################
include(cmake/SetupDependencies.cmake)

add_library(tb-eigen1 OBJECT source/benchmark/tensor_product_eigen1.cpp)
add_library(tb-eigen2 OBJECT source/benchmark/tensor_product_eigen2.cpp)
add_library(tb-eigen3 OBJECT source/benchmark/tensor_product_eigen3.cpp)
add_library(tb-xtensor OBJECT source/benchmark/tensor_product_xtensor.cpp)
add_library(tb-tblis OBJECT source/benchmark/tensor_product_tblis.cpp)
add_library(tb-cyclops OBJECT source/benchmark/tensor_product_cyclops.cpp)
if(TB_ENABLE_CUTENSOR)
    enable_language(CUDA)
    add_library(tb-cutensor OBJECT source/benchmark/tensor_product_cutensor.cu)
else()
    # Add this dummy that can be compiled without enabling CUDA
    add_library(tb-cutensor OBJECT source/benchmark/tensor_product_cutensor.cpp)
endif()



target_link_libraries(tb-eigen1 PRIVATE tb-flags tb-deps)
target_link_libraries(tb-eigen2 PRIVATE tb-flags tb-deps)
target_link_libraries(tb-eigen3 PRIVATE tb-flags tb-deps)
target_link_libraries(tb-xtensor PRIVATE tb-flags tb-deps)
target_link_libraries(tb-tblis PRIVATE tb-flags tb-deps)
target_link_libraries(tb-cyclops PRIVATE tb-flags tb-deps)
target_link_libraries(tb-cutensor PRIVATE tb-flags tb-deps)

target_include_directories(tb-eigen1 PRIVATE source)
target_include_directories(tb-eigen2 PRIVATE source)
target_include_directories(tb-eigen3 PRIVATE source)
target_include_directories(tb-xtensor PRIVATE source)
target_include_directories(tb-tblis PRIVATE source)
target_include_directories(tb-cyclops PRIVATE source)
target_include_directories(tb-cutensor PRIVATE source)

if(TB_ENABLE_EIGEN1)
    target_compile_definitions(tb-eigen1 PUBLIC TB_EIGEN1)
endif()
if(TB_ENABLE_EIGEN2)
    target_compile_definitions(tb-eigen2 PUBLIC TB_EIGEN2)
endif()
if(TB_ENABLE_EIGEN3)
    target_compile_definitions(tb-eigen3 PUBLIC TB_EIGEN3)
endif()
if(TB_ENABLE_CUTENSOR)
    target_compile_definitions(tb-cutensor PUBLIC TB_CUTENSOR TB_CUDA)
    target_link_libraries(tb-cutensor INTERFACE cuda cutensor cublas cublasLt)
endif()
if(TB_ENABLE_XTENSOR)
    target_compile_definitions(tb-xtensor PUBLIC TB_XTENSOR)
endif()
if(TB_ENABLE_TBLIS)
    target_compile_definitions(tb-tblis PUBLIC TB_TBLIS)
endif()
if(TB_ENABLE_CYCLOPS)
    target_compile_definitions(tb-cyclops PUBLIC TB_CYCLOPS)
endif()

##################################################################
### Link all the things!                                       ###
##################################################################
target_link_libraries(tensorbench PRIVATE tb-eigen1)
target_link_libraries(tensorbench PRIVATE tb-eigen2)
target_link_libraries(tensorbench PRIVATE tb-eigen3)
target_link_libraries(tensorbench PRIVATE tb-xtensor)
target_link_libraries(tensorbench PRIVATE tb-tblis)
target_link_libraries(tensorbench PRIVATE tb-cyclops)
target_link_libraries(tensorbench PRIVATE tb-cutensor)
target_link_libraries(tensorbench PRIVATE tb-deps tb-flags)  # Link everything to main project

if (TARGET mold::mold)
    target_link_libraries(tensorbench PRIVATE mold::mold)
endif()

# Print summary of CMake configuration
if (TB_CMAKE_DEBUG)
    # Print summary of CMake configuration
    include(cmake/PrintTargetInfo.cmake)
    print_and_write_project_summary(tensorbench)
endif()

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TBUninstall.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/TBUninstall.cmake
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/TBUninstall.cmake)
endif()

