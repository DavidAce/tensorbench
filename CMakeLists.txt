cmake_minimum_required(VERSION 3.20)
project(tensorbench
        VERSION 0.0.1
        DESCRIPTION "Benchmarks for tensor operations"
        HOMEPAGE_URL "")


# Make sure we use DMRG's own find modules
list(INSERT CMAKE_MODULE_PATH 0  ${PROJECT_SOURCE_DIR}/cmake)

# Set options this build
option(BUILD_SHARED_LIBS           "Build shared libraries"                                       OFF)
option(TB_PREFER_CONDA_LIBS        "Prefer libraries installed through anaconda"                  OFF)
option(TB_ENABLE_OPENMP            "Use OpenMP"                                                   ON)
option(TB_ENABLE_MKL               "Enable Intel Math Kernel Library"                             OFF)
option(TB_ENABLE_OPENBLAS          "Enable OpenBLAS"                                              OFF)
option(TB_EIGEN3_BLAS              "Enable BLAS backend for Eigen3 (if OpenBLAS/MKL is enabled)"  OFF)
option(TB_ENABLE_EIGEN1            "Benchmark contractions on CPU with Eigen mode 1"              OFF)
option(TB_ENABLE_EIGEN2            "Benchmark contractions on CPU with Eigen mode 2"              OFF)
option(TB_ENABLE_EIGEN3            "Benchmark contractions on CPU with Eigen mode 3"              OFF)
option(TB_ENABLE_CUDA              "Benchmark contractions on GPU with Eigen GPU routines"        OFF)
option(TB_ENABLE_ACRO              "Benchmark contractions on GPU with acrotensor"                OFF)
option(TB_ENABLE_CUTE              "Benchmark contractions on GPU with CUTENSOR"                  OFF)
option(TB_ENABLE_XTENSOR           "Benchmark contractions on CPU with xtensor"                   OFF)
option(TB_ENABLE_TBLIS             "Benchmark contractions on CPU with TBLIS"                     OFF)
option(TB_ENABLE_CYCLOPS           "Benchmark contractions on CPU with CYCLOPS"                   OFF)
option(TB_ENABLE_ASAN              "Enable sanitizer -fsanitize=address"                          OFF)
option(TB_ENABLE_USAN              "Enable sanitizer -fsanitize=undefined"                        OFF)
option(TB_ENABLE_LTO               "Enable link time optimization"                                ON)
option(TB_ENABLE_TESTS             "Enable unit testing with ctest"                               OFF)
option(TB_BUILD_EXAMPLES           "Build examples"                                               OFF)
option(TB_PRINT_INFO               "Print information during cmake configure"                     OFF)
option(TB_PRINT_CHECKS             "Print compilation check details during cmake configure"       OFF)
option(TB_DEPS_IN_SUBDIR           "Install dependencies into CMAKE_INSTALL_PREFIX/<libname>"     ON)

if(TB_ENABLE_OPENBLAS AND TB_ENABLE_MKL)
    message(FATAL_ERROR "MKL and OPENBLAS are mutually exclusive")
endif()

# Make an "enum" for valid download methods
set(TB_PACKAGE_MANAGERS_VALID find cmake find-or-cmake conan)
set(TB_PACKAGE_MANAGER find CACHE STRING "Download method for external dependencies")
set_property(CACHE TB_PACKAGE_MANAGER PROPERTY STRINGS ${TB_PACKAGE_MANAGERS_VALID})
if (NOT TB_PACKAGE_MANAGER IN_LIST TB_PACKAGE_MANAGERS_VALID)
    message(FATAL_ERROR "TB_PACKAGE_MANAGER must be one of ${TB_PACKAGE_MANAGERS_VALID}")
endif ()


# Setup paths that find_package should search and
# let cmake find our Find<package>.cmake modules
include(cmake/SetupPaths.cmake)

################################################################
### Get git version number                                   ###
### Generates a header gitversion/gitversion.h               ###
### Include it using #include <gitversion.h>                 ###
### Gives a namespace GIT:: with several git version numbers.###
################################################################
include(cmake/gitversion.cmake)



############################################################
### Print extra info during CMake configure              ###
############################################################
include(cmake/PrintHostInfo.cmake)
include(cmake/PrintOptionInfo.cmake)

###########################################
###  Apply RELEASE/DEBUG compile flags  ###
###########################################
include(cmake/CompilerFlags.cmake)



################################
###  Add all source files    ###
################################
add_executable(tensorbench
        source/main.cpp
        source/tools/prof.cpp
        source/tools/log.cpp
        source/tools/class_tic_toc.cpp
        source/math/tenx/omp.cpp
        source/mpi/mpi-tools.cpp
#        source/tid/tid.cpp
#        source/tid/token.cpp
#        source/tid/ur.cpp
        source/storage/results.cpp)
target_include_directories(tensorbench PRIVATE source)
set_target_properties(tensorbench PROPERTIES LINK_WHAT_YOU_USE TRUE)
target_enable_lto(tensorbench)

###########################################
### Fetch or find all the dependencies  ###
###########################################
include(cmake/SetupDependencies.cmake)


##################################################################
### Link all the things!                                       ###
##################################################################

if(TB_ENABLE_EIGEN1)
    add_library(tb-eigen1 OBJECT source/contract/tensor_product_eigen1.cpp)
    target_include_directories(tb-eigen1 PRIVATE source)
    target_compile_definitions(tb-eigen1 PUBLIC TB_EIGEN1)
    target_link_libraries(tb-eigen1 PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-eigen1)
    if(TB_EIGEN3_BLAS)
        target_compile_definitions(tb-eigen1 PRIVATE EIGEN_USE_BLAS)
        target_link_libraries(tb-eigen1 PRIVATE BLAS::BLAS)
    endif()
endif()
if(TB_ENABLE_EIGEN2)
    add_library(tb-eigen2 OBJECT source/contract/tensor_product_eigen2.cpp)
    target_include_directories(tb-eigen2 PRIVATE source)
    target_compile_definitions(tb-eigen2 PUBLIC TB_EIGEN2)
    target_link_libraries(tb-eigen2 PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-eigen2)
    if(TB_EIGEN3_BLAS)
        target_compile_definitions(tb-eigen2 PRIVATE EIGEN_USE_BLAS)
        target_link_libraries(tb-eigen2 PRIVATE BLAS::BLAS)
    endif()
endif()
if(TB_ENABLE_EIGEN3)
    add_library(tb-eigen3 OBJECT source/contract/tensor_product_eigen3.cpp)
    target_include_directories(tb-eigen3 PRIVATE source)
    target_compile_definitions(tb-eigen3 PUBLIC TB_EIGEN3)
    target_link_libraries(tb-eigen3 PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-eigen3)
    if(TB_EIGEN3_BLAS)
        target_compile_definitions(tb-eigen3 PRIVATE EIGEN_USE_BLAS)
        target_link_libraries(tb-eigen3 PRIVATE BLAS::BLAS)
    endif()
endif()

if(TB_ENABLE_CUTE)
    enable_language(CUDA)
    add_library(tb-cute OBJECT source/contract/tensor_product_cute.cu)
    target_include_directories(tb-cute PRIVATE source)
    target_compile_definitions(tb-cute PUBLIC TB_CUTE)
    target_link_libraries(tb-cute INTERFACE cuda cutensor)
    target_link_libraries(tb-cute PRIVATE tb-deps tb-flags)
    target_link_libraries(tensorbench PRIVATE tb-cute)
endif()

if(TB_ENABLE_XTENSOR)
    add_library(tb-xtensor OBJECT source/contract/tensor_product_xtensor.cpp)
    target_include_directories(tb-xtensor PRIVATE source)
    target_compile_definitions(tb-xtensor PUBLIC TB_XTENSOR)
    target_link_libraries(tb-xtensor PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-xtensor)
endif()

if(TB_ENABLE_TBLIS)
    add_library(tb-tblis OBJECT source/contract/tensor_product_tblis.cpp)
    target_include_directories(tb-tblis PRIVATE source)
    target_compile_definitions(tb-tblis PUBLIC TB_TBLIS)
    target_link_libraries(tb-tblis PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-tblis)
endif()
if(TB_ENABLE_CYCLOPS)
    add_library(tb-cyclops OBJECT source/contract/tensor_product_cyclops.cpp)
    target_include_directories(tb-cyclops PRIVATE source)
    target_compile_definitions(tb-cyclops PUBLIC TB_CYCLOPS)
    target_link_libraries(tb-cyclops PRIVATE tb-flags tb-deps)
    target_link_libraries(tensorbench PRIVATE tb-cyclops)
endif()



add_subdirectory(source/tid)
target_link_libraries(tensorbench PRIVATE tb-deps tb-flags)  # Link everything to main project
target_link_libraries(tensorbench PRIVATE tb-tid-o)

# Print summary of CMake configuration
if (TB_PRINT_INFO)
    include(cmake/PrintTargetInfo.cmake)
    include(cmake/getExpandedTarget.cmake)
    expand_target_all_targets(tb-deps TB_DEPS_EXPANDED)
    message(STATUS "| TB TARGET SUMMARY")
    message(STATUS "|--------------------")
    print_compiler_info("| ")
    print_target_info(tensorbench "| ")
    foreach(tgt tb-flags;${TB_DEPS_EXPANDED})
        print_target_info(${tgt} "| ")
    endforeach()
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

