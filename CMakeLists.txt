cmake_minimum_required(VERSION 3.12)
project(tensorbench
        VERSION 0.0.1
        DESCRIPTION "Benchmarks for tensor operations"
        HOMEPAGE_URL "")


# Make sure we use DMRG's own find modules
list(INSERT CMAKE_MODULE_PATH 0  ${PROJECT_SOURCE_DIR}/cmake-modules)

# Set options this build
option(TB_PREFER_CONDA_LIBS        "Prefer libraries installed through anaconda"                  OFF)
option(TB_ENABLE_OPENMP            "Use OpenMP"                                                   ON)
option(TB_ENABLE_MKL               "Enable Intel Math Kernel Library"                             OFF)
option(TB_EIGEN3_BLAS              "Enable BLAS backend for Eigen3 (if OpenBLAS/MKL is enabled)"  OFF)
option(TB_ENABLE_LTO               "Enable link time optimization"                                OFF)
option(TB_ENABLE_TESTS             "Enable unit testing with ctest"                               OFF)
option(TB_BUILD_EXAMPLES           "Build examples"                                               OFF)
option(TB_PRINT_INFO               "Print information during cmake configure"                     OFF)
option(TB_DEPS_IN_SUBDIR           "Install dependencies into CMAKE_INSTALL_PREFIX/<libname>"     ON)
option(TB_ENABLE_ASAN              "Enable runtime address sanitizer -fsanitize=address"          OFF)
option(TB_ENABLE_CPU1              "Benchmark contractions on CPU mode 1"                         OFF)
option(TB_ENABLE_CPU2              "Benchmark contractions on CPU mode 2"                         OFF)
option(TB_ENABLE_CPU3              "Benchmark contractions on CPU mode 3"                         OFF)
option(TB_ENABLE_CUDA              "Benchmark contractions on GPU with Eigen Tensor GPU routines" OFF)
option(TB_ENABLE_ACRO              "Benchmark contractions on GPU with acrotensor"                OFF)
option(TB_ENABLE_CUTE              "Benchmark contractions on GPU with CUTENSOR"                  OFF)


# Make an "enum" for valid download methods
set(TB_DOWNLOAD_METHODS_VALID find fetch find-or-fetch conan)
set(TB_DOWNLOAD_METHOD find CACHE STRING "Download method for external dependencies")
set_property(CACHE TB_DOWNLOAD_METHOD PROPERTY STRINGS ${TB_DOWNLOAD_METHODS_VALID})
if (NOT TB_DOWNLOAD_METHOD IN_LIST TB_DOWNLOAD_METHODS_VALID)
    message(FATAL_ERROR "TB_DOWNLOAD_METHOD must be one of ${TB_DOWNLOAD_METHODS_VALID}")
endif ()




# Set default policies if CMake is new enough
if (CMAKE_VERSION VERSION_LESS 3.12)
    message(STATUS "Not setting policies")
else()
    cmake_policy(SET CMP0074 NEW) # Use <PackageName>_ROOT variables in find_package calls
    cmake_policy(SET CMP0075 NEW)
    # Use cmake_policy CMP0067 to let try_compile honor current CMAKE_CXX_STANDARD setting.
    # See here https://stackoverflow.com/questions/47213356/cmake-using-corrext-c-standard-when-checking-for-header-files
    cmake_policy(SET CMP0067 NEW)
endif()



################################################################
### Get git version number                                   ###
### Generates a header gitversion/gitversion.h               ###
### Include it using #include <gitversion.h>                 ###
### Gives a namespace GIT:: with several git version numbers.###
################################################################
include(cmake-modules/gitversion.cmake)


##################################################################
### Setup dependency paths                                     ###
##################################################################
include(GNUInstallDirs)
string(TOLOWER ${CMAKE_BUILD_TYPE} cmake_build_type)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(STATUS "Setting default install prefix -- CMAKE_INSTALL_PREFIX --> ${CMAKE_BINARY_DIR}/tb-deps-install")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/tb-deps-install" CACHE PATH "..." FORCE)
endif()



############################################################
### Print extra info during CMake configure              ###
############################################################
include(cmake-modules/PrintHostInfo.cmake)
include(cmake-modules/PrintOptionInfo.cmake)

###########################################
###  Apply RELEASE/DEBUG compile flags  ###
###########################################
include(cmake-modules/CompilerFlags.cmake)



################################
###  Add all source files    ###
################################
add_executable(${PROJECT_NAME}
        source/main.cpp
        source/tools/prof.cpp
        source/tools/log.cpp
        source/tools/class_tic_toc.cpp
        source/storage/results.cpp source/storage/results.h)
target_include_directories(${PROJECT_NAME} PRIVATE source)
add_library(project-settings INTERFACE)
##################################################################
###  Force cmake to find .a library suffixes on static builds  ###
##################################################################
if(BUILD_SHARED_LIBS)
    set(LINK_TYPE SHARED)
else()
    set(LINK_TYPE STATIC)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} )
#    target_link_libraries  (project-settings INTERFACE -static)   ### Static linkage
endif()




##############################
###  Enable c++17 support  ###
##############################
#target_compile_features(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:cxx_std_17>)
#target_compile_features(project-settings INTERFACE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)


#######################################################
###  Check  #include<optional> or                   ###
###  or     #include<experimental/optional>         ###
#######################################################
include(${PROJECT_SOURCE_DIR}/cmake-modules/CheckCXXOptional.cmake)
CheckCXXOptional()



##################################################
###  Add optional RELEASE/DEBUG compile flags  ###
##################################################
# -fsanitize=leak -fsanitize=undefined -fsanitize=alignment  -fsanitize=leak -fsanitize=undefined -fsanitize=alignment
target_compile_options(project-settings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-Wall>)
target_compile_options(project-settings INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
target_compile_options(project-settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELEASE>>:-g  -mfma>)
target_compile_options(project-settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:DEBUG>>: -Wpedantic -Wextra -Wconversion>)
target_compile_options(project-settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:RELWITHDEBINFO>>:>)
target_compile_options(project-settings INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:MINSIZEREL>>:>)



#######################################
# Settings for sanitizers           ###
#######################################
if(TB_ENABLE_ASAN)
    target_compile_options(project-settings INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(project-settings INTERFACE -fsanitize=address)
endif()

#######################################
### Compiler-dependent linker flags ###
#######################################
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" )
    target_link_libraries(project-settings INTERFACE  -stdlib=libstdc++ )
endif()


##################################################################
### Check if we can enable Link Time Optimization              ###
##################################################################
if(TB_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT supported OUTPUT error)
    if(supported)
        message(STATUS "IPO / LTO supported, enabling")
        set_target_properties(${PROJECT_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
    else()
        message(STATUS "IPO / LTO not supported: <${error}>, will not enable")
        set(TB_ENABLE_LTO OFF CACHE)
    endif()
endif()





#################################################################
### Prepend conda dirs for hints when using find_package      ###
#################################################################
if(TB_PREFER_CONDA_LIBS)
    list(APPEND CONDA_HINTS
            $ENV{CONDA_PREFIX}
            $ENV{HOME}/anaconda3
            $ENV{HOME}/miniconda
            )
endif()


##################################################################
### Preempt Threads::Threads                                   ###
### It's looked for in dependencies, so we make it right       ###
### before it's done wrong, i.e. with pthread instead of       ###
### -lpthread.                                     ###
##################################################################
if(NOT TARGET Threads::Threads)
    set(CMAKE_THREAD_PREFER_PTHREAD FALSE)
    set(THREADS_PREFER_PTHREAD_FLAG FALSE)
    find_package(Threads)
    set_target_properties(Threads::Threads PROPERTIES INTERFACE_LINK_LIBRARIES pthread)
endif()



###########################################
### Fetch or find all the dependencies  ###
###########################################
include(cmake-modules/SetupDependenciesFind.cmake)
include(cmake-modules/SetupDependenciesFetch.cmake)
include(cmake-modules/SetupDependenciesConan.cmake)

##################################################################
### Link all the things!                                       ###
##################################################################
#target_link_libraries(project-settings INTERFACE /home/david/.conan/data/fmt/6.0.0/bincrafters/stable/package/83b377af2053f637540e812f25798c29f53e4921/lib/libfmt.a)
#
target_link_libraries(project-settings INTERFACE  ${CONAN_TARGETS} ${FOUND_TARGETS}) # Order is important here

if(NOT TARGET openmp::openmp)
    target_compile_options(project-settings INTERFACE -Wno-unknown-pragmas)
endif()

target_link_libraries(project-settings INTERFACE -Wl,--whole-archive pthread -Wl,--no-whole-archive -lrt -ldl )
target_link_libraries(${PROJECT_NAME} PRIVATE project-settings )  # Link everything to main project


if(TB_ENABLE_CPU1)
    add_library(tb-cpu1 OBJECT source/contract/hamiltonian_squared_dot_psi_v1.cpp)
    target_include_directories(tb-cpu1 PRIVATE source)
    target_compile_definitions(tb-cpu1 PUBLIC TB_CPU1)
    target_compile_options(tb-cpu1 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-cpu1 PRIVATE project-settings)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-cpu1)
endif()
if(TB_ENABLE_CPU2)
    add_library(tb-cpu2 OBJECT source/contract/hamiltonian_squared_dot_psi_v2.cpp)
    target_include_directories(tb-cpu2 PRIVATE source)
    target_compile_definitions(tb-cpu2 PUBLIC TB_CPU2)
    target_compile_options(tb-cpu2 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-cpu2 PRIVATE project-settings)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-cpu2)
endif()
if(TB_ENABLE_CPU3)
    add_library(tb-cpu3 OBJECT source/contract/hamiltonian_squared_dot_psi_v3.cpp)
    target_include_directories(tb-cpu3 PRIVATE source)
    target_compile_definitions(tb-cpu3 PUBLIC TB_CPU3)
    target_compile_options(tb-cpu3 PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>)
    target_link_libraries(tb-cpu3 PRIVATE project-settings)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-cpu3)
endif()

if(TB_ENABLE_CUDA)
    enable_language(CUDA)
    add_library(tb-cuda OBJECT source/contract/hamiltonian_squared_dot_psi_cuda.cu)
    target_include_directories(tb-cuda PRIVATE source)
    target_compile_options(tb-cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(tb-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
    target_compile_options(tb-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-std=c++17 >)
    target_compile_options(tb-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    target_compile_options(tb-cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler>)
    target_compile_options(tb-cuda PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
    target_link_libraries(tb-cuda PRIVATE project-settings)
    target_compile_definitions(tb-cuda PUBLIC EIGEN_USE_GPU)
    target_compile_definitions(tb-cuda PUBLIC TB_CUDA)
#    target_compile_definitions(tb-cuda PUBLIC EIGEN_MALLOC_ALREADY_ALIGNED=0) # May work to fix CERES segfaults!!!
#    target_compile_definitions(tb-cuda PUBLIC EIGEN_MAX_ALIGN_BYTES=32)
    set_target_properties(tb-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(tb-cuda PROPERTIES CUDA_PTX_COMPILATION OFF)
    set_target_properties(tb-cuda PROPERTIES CUDA_ARCHITECTURES 75)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-cuda)
endif()

if(TB_ENABLE_ACRO)
    enable_language(CUDA)
    add_library(tb-acro OBJECT source/contract/hamiltonian_squared_dot_psi_acro.cu)
    target_include_directories(tb-acro PRIVATE source)
    target_compile_options(tb-acro PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(tb-acro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
    target_compile_options(tb-acro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-std=c++17 >)
    target_compile_options(tb-acro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    target_compile_options(tb-acro PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler>)
    target_compile_options(tb-acro PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
    target_compile_definitions(tb-acro PUBLIC TB_ACRO)
    target_link_libraries(tb-acro PRIVATE project-settings)
    target_link_libraries(tb-acro INTERFACE cuda nvrtc)
    set_target_properties(tb-acro PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(tb-acro PROPERTIES CUDA_PTX_COMPILATION OFF)
    set_target_properties(tb-acro PROPERTIES CUDA_ARCHITECTURES 75)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-acro)
endif()

if(TB_ENABLE_CUTE)
    enable_language(CUDA)
    add_library(tb-cute OBJECT source/contract/hamiltonian_squared_dot_psi_cute.cu)
    target_include_directories(tb-cute PRIVATE source)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-std=c++17 >)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr>)
    target_compile_options(tb-cute PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -forward-unknown-to-host-compiler>)
    target_compile_options(tb-cute PUBLIC $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)
    target_compile_definitions(tb-cute PUBLIC TB_CUTE)
    target_link_libraries(tb-cute INTERFACE cuda cutensor)
    target_link_libraries(tb-cute PRIVATE project-settings)
    set_target_properties(tb-cute PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    set_target_properties(tb-cute PROPERTIES CUDA_PTX_COMPILATION OFF)
    set_target_properties(tb-cute PROPERTIES CUDA_ARCHITECTURES 75)
    target_link_libraries(${PROJECT_NAME} PRIVATE tb-cute)
endif()

if(TB_ENABLE_CUDA OR TB_ENABLE_ACRO OR TB_ENABLE_CUTE)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES 75)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -std=c++17>)
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -std=c++17>)
endif()

# Speed up with precompiled headers
if(COMMAND target_precompile_headers)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <spdlog/spdlog.h>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <fmt/core.h> <fmt/format.h> <fmt/ostream.h> <fmt/ranges.h>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <general/nmspc_tensor_extra.h>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <Eigen/Dense>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <Eigen/Core>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <unsupported/Eigen/CXX11/Tensor>)
    target_precompile_headers(${PROJECT_NAME} PRIVATE <vector> <array> <complex> <map> <optional> <iostream> <iomanip> <memory>)
#
#    if (TARGET tb-cpu)
#        target_precompile_headers(tb-cpu PRIVATE <spdlog/spdlog.h>)
#        target_precompile_headers(tb-cpu PRIVATE <fmt/core.h> <fmt/format.h> <fmt/ostream.h> <fmt/ranges.h>)
#        target_precompile_headers(tb-cpu PRIVATE <general/nmspc_tensor_extra.h>)
#        target_precompile_headers(tb-cpu PRIVATE <Eigen/Dense>)
#        target_precompile_headers(tb-cpu PRIVATE <Eigen/Core>)
#        target_precompile_headers(tb-cpu PRIVATE <unsupported/Eigen/CXX11/Tensor>)
#        target_precompile_headers(tb-cpu PRIVATE <vector> <array> <complex> <map> <optional> <iostream> <iomanip> <memory>)
#    endif()
#    if (TARGET tb-cuda)
#        target_precompile_headers(tb-cuda REUSE_FROM ${PROJECT_NAME})
#    endif()
#    if (TARGET tb-acro)
#        target_precompile_headers(tb-acro REUSE_FROM ${PROJECT_NAME})
#    endif()
#    if (TARGET tb-cute)
#        target_precompile_headers(tb-cute REUSE_FROM ${PROJECT_NAME})
#    endif()
endif()


# Print summary of CMake configuration
if (TB_PRINT_INFO)
    include(cmake-modules/PrintTargetInfo.cmake)
    include(cmake-modules/getExpandedTarget.cmake)
    expand_target_all_targets(project-settings TB_TARGETS_EXPANDED)
    if(TB_ENABLE_CPU)
        list(APPEND TB_TARGETS_EXPANDED tb-cpu)
    endif()
    if(TB_ENABLE_CUDA)
        list(APPEND TB_TARGETS_EXPANDED tb-cuda)
    endif()
    if(TB_ENABLE_ACRO)
        list(APPEND TB_TARGETS_EXPANDED tb-acro)
    endif()
    if(TB_ENABLE_CUTE)
        list(APPEND TB_TARGETS_EXPANDED tb-cute)
    endif()
    message(STATUS "| TB TARGET SUMMARY")
    message(STATUS "|--------------------")
    print_compiler_info("| ")
    foreach(tgt ${TB_TARGETS_EXPANDED})
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
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules/TBUninstall.cmake.in
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





