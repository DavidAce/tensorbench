cmake_minimum_required(VERSION 3.12)
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
option(TB_ENABLE_PCH               "Speed up compilation with precompiled headers"                OFF)
option(TB_ENABLE_CCACHE            "Speed up compilation with ccache"                             OFF)
option(TB_ENABLE_ASAN              "Enable sanitizer -fsanitize=address"                          OFF)
option(TB_ENABLE_USAN              "Enable sanitizer -fsanitize=undefined"                        OFF)
option(TB_ENABLE_LTO               "Enable link time optimization"                                OFF)
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
        source/storage/results.cpp source/storage/results.h)
target_include_directories(tensorbench PRIVATE source)
set_target_properties(tensorbench PROPERTIES LINK_WHAT_YOU_USE TRUE)

add_library(tb-flags INTERFACE)


##############################
###  Enable c++17 support  ###
##############################
#target_compile_features(tensorbench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:cxx_std_17>)
#target_compile_features(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
target_compile_options(tensorbench PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)


#######################################################
###  Check  #include<optional> or                   ###
###  or     #include<experimental/optional>         ###
#######################################################
include(${PROJECT_SOURCE_DIR}/cmake/CheckCXXOptional.cmake)
CheckCXXOptional()



##################################################
###  Add optional RELEASE/DEBUG compile flags  ###
##################################################
# -fsanitize=leak -fsanitize=undefined -fsanitize=alignment  -fsanitize=leak -fsanitize=undefined -fsanitize=alignment
target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)




#######################################
# Settings for sanitizers           ###
#######################################
if(TB_ENABLE_ASAN)
    target_compile_options(tb-flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(tb-flags INTERFACE -fsanitize=address)
endif()

#######################################
### Compiler-dependent linker flags ###
#######################################
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
    target_link_libraries(tb-flags INTERFACE  -stdlib=libstdc++ )
endif()


##################################################################
### Check if we can enable Link Time Optimization              ###
##################################################################
if(TB_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT supported OUTPUT error)
    if(supported)
        message(STATUS "IPO / LTO supported, enabling")
        set_target_properties(DMRG++ PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        if(CMAKE_VERSION VERSION_GREATER 3.19)
            set_target_properties(tb-flags PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        endif()
    else()
        message(STATUS "IPO / LTO not supported: <${error}>, will not enable")
        set(TB_ENABLE_LTO OFF CACHE)
    endif ()
endif ()






###########################################
### Fetch or find all the dependencies  ###
###########################################
include(cmake/SetupDependencies.cmake)


##################################################################
### Link all the things!                                       ###
##################################################################
#target_link_libraries(tb-flags INTERFACE /home/david/.conan/data/fmt/6.0.0/bincrafters/stable/package/83b377af2053f637540e812f25798c29f53e4921/lib/libfmt.a)
#
#target_link_libraries(tb-flags INTERFACE  ${CONAN_TARGETS} ${FOUND_TARGETS}) # Order is important here

if(NOT TARGET openmp::openmp)
    target_compile_options(tb-flags INTERFACE -Wno-unknown-pragmas)
endif()

if(TB_ENABLE_EIGEN1)
    add_library(tb-eigen1 OBJECT source/contract/tensor_product_eigen1.cpp)
    target_include_directories(tb-eigen1 PRIVATE source)
    target_compile_definitions(tb-eigen1 PUBLIC TB_EIGEN1)
    target_compile_options(tb-eigen1 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-eigen1 PRIVATE tb-flags tb-deps)
    list(APPEND tb-extra tb-eigen1)
endif()
if(TB_ENABLE_EIGEN2)
    add_library(tb-eigen2 OBJECT source/contract/tensor_product_eigen2.cpp)
    target_include_directories(tb-eigen2 PRIVATE source)
    target_compile_definitions(tb-eigen2 PUBLIC TB_EIGEN2)
    target_compile_options(tb-eigen2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-eigen2 PRIVATE tb-flags tb-deps)
    list(APPEND tb-extra tb-eigen2)
endif()
if(TB_ENABLE_EIGEN3)
    add_library(tb-eigen3 OBJECT source/contract/tensor_product_eigen3.cpp)
    target_include_directories(tb-eigen3 PRIVATE source)
    target_compile_definitions(tb-eigen3 PUBLIC TB_EIGEN3)
    target_compile_options(tb-eigen3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-eigen3 PRIVATE tb-flags tb-deps)
    list(APPEND tb-extra tb-eigen3)
endif()

if(TB_ENABLE_CUTE)
    enable_language(CUDA)
    add_library(tb-cute OBJECT source/contract/tensor_product_cute.cu)
    target_include_directories(tb-cute PRIVATE source)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-std=c++17 >)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler>)
    target_compile_options(tb-cute PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
    target_compile_definitions(tb-cute PUBLIC TB_CUTE)
    target_link_libraries(tb-cute INTERFACE cuda cutensor)
    target_link_libraries(tb-cute PRIVATE tb-flags tb-deps)
    set_target_properties(tb-cute PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(tb-cute PROPERTIES CUDA_PTX_COMPILATION OFF)
    set_target_properties(tb-cute PROPERTIES CUDA_ARCHITECTURES 75)
    list(APPEND tb-extra tb-cute)
endif()

if(TB_ENABLE_CUTE)
    set_target_properties(tensorbench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(tensorbench PROPERTIES CUDA_ARCHITECTURES 75)
    target_compile_options(tensorbench PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(tensorbench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
endif()

if(TB_ENABLE_XTENSOR)
    add_library(tb-xtensor OBJECT source/contract/tensor_product_xtensor.cpp)
    target_include_directories(tb-xtensor PRIVATE source)
    target_compile_definitions(tb-xtensor PUBLIC TB_XTENSOR)
    target_compile_options(tb-xtensor PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-xtensor PRIVATE tb-flags tb-deps)
    list(APPEND tb-extra tb-xtensor)
endif()

if(TB_ENABLE_TBLIS)
    add_library(tb-tblis OBJECT source/contract/tensor_product_tblis.cpp)
    target_include_directories(tb-tblis PRIVATE source)
    target_compile_definitions(tb-tblis PUBLIC TB_TBLIS)
    target_link_libraries(tb-tblis PRIVATE tb-flags tb-deps)
    list(APPEND tb-extra tb-tblis)
endif()



# Speed up with precompiled headers
if(COMMAND target_precompile_headers)
    target_precompile_headers(tensorbench PRIVATE <spdlog/spdlog.h>)
    target_precompile_headers(tensorbench PRIVATE <fmt/core.h> <fmt/format.h> <fmt/ostream.h> <fmt/ranges.h>)
    target_precompile_headers(tensorbench PRIVATE <Eigen/Dense>)
    target_precompile_headers(tensorbench PRIVATE <Eigen/Core>)
    target_precompile_headers(tensorbench PRIVATE <unsupported/Eigen/CXX11/Tensor>)
    target_precompile_headers(tensorbench PRIVATE <vector> <array> <complex> <map> <optional> <iostream> <iomanip> <memory>)
endif()


add_subdirectory(source/tid)
#target_link_libraries(tb-flags INTERFACE -Wl,--whole-archive pthread -Wl,--no-whole-archive rt dl )
target_link_libraries(tensorbench PRIVATE tb-tid ${tb-extra} tb-deps tb-flags)  # Link everything to main project



# Print summary of CMake configuration
if (TB_PRINT_INFO)
    include(cmake/PrintTargetInfo.cmake)
    include(cmake/getExpandedTarget.cmake)
    expand_target_all_targets(tb-deps TB_TARGETS_EXPANDED)
    message(STATUS "| TB TARGET SUMMARY")
    message(STATUS "|--------------------")
    print_compiler_info("| ")
    print_target_info(tensorbench "| ")
    foreach(tgt ${TB_TARGETS_EXPANDED};${tb-extra})
        print_target_info(${tgt} "| ")
    endforeach()
endif()



###################
### INFORMATION ###
###################
message("")
message("=============================================================================")
message("")
message("      To clean cmake files, run './build.sh -c'")
message("      To see more options,  run './build.sh -h'")
message("")
message("=============================================================================")
message("")


# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TBUninstall.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/TBUninstall.cmake
            IMMEDIATE @ONLY)

    add_custom_target(uninstall
            COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/TBUninstall.cmake)
endif()

#############################
## UNIT TEST
#############################
# Simple testing with ctest
#if (TB_ENABLE_TESTS)
#    enable_testing()
#    add_subdirectory(tests)
#endif ()





