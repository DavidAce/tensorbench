# Append search paths for find_package and find_library calls
include(${PROJECT_SOURCE_DIR}/cmake/cmake_dependency_provider/PKGInstall.cmake)
list(PREPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" CACHE PATH "find_package module dir" FORCE)

if(NOT TARGET tb-deps)
    add_library(tb-deps INTERFACE)
endif()

find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS CXX REQUIRED)


find_package(Eigen3     3.4.0  REQUIRED)    # Eigen3 numerical library
find_package(h5pp       1.11.0 REQUIRED)    # Writing to file binary HDF5 format
find_package(fmt        10.2.0  REQUIRED)    # String formatter
find_package(spdlog     1.13.0 REQUIRED)    # Fast logger
find_package(CLI11      2.4.1  REQUIRED)    # Command line argument parser
find_package(Backward   1.6    REQUIRED)    # Pretty stack traces


##################################################################
### Link all the things!                                       ###
##################################################################
target_link_libraries(tb-flags INTERFACE OpenMP::OpenMP_CXX)
target_compile_definitions(tb-flags INTERFACE EIGEN_USE_THREADS) # For Eigen::Tensor parallelization
target_link_libraries(tb-deps INTERFACE
                      h5pp::h5pp
                      Eigen3::Eigen
                      fmt::fmt
                      spdlog::spdlog
                      CLI11::CLI11
                      Backward::Backward)


# Additional optional libraries

if (TB_ENABLE_MKL)
    if(TB_ENABLE_CYCLOPS)
        find_package(MPI COMPONENTS CXX REQUIRED)
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 scalapack blacs_openmpi REQUIRED)
    else()
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 REQUIRED)
    endif()
    target_compile_definitions(mkl::mkl INTERFACE TB_MKL)
endif ()
find_package(BLAS REQUIRED)

if(TB_ENABLE_TBLIS)
    pkg_install(tblis)
    find_package(tblis REQUIRED MODULE)
    target_link_libraries(tb-deps INTERFACE tblis::tblis)
endif()

if(TB_ENABLE_XTENSOR)
    find_package(xtensor REQUIRED) # Given by pkg or conan already
    pkg_install(xtensor-blas)  # Does not exist in conan
    find_package(xtensor-blas REQUIRED)
    target_compile_definitions(xtensor-blas INTERFACE HAVE_CBLAS=1)
    target_link_libraries(xtensor INTERFACE xtensor-blas)
    target_link_libraries(tb-deps INTERFACE xtensor)
endif()

if(TB_ENABLE_CYCLOPS)
    find_package(MPI COMPONENTS CXX REQUIRED)
    find_package(gfortran REQUIRED)
    pkg_install(hptt)
    find_package(hptt REQUIRED)
    pkg_install(scalapack)
    find_package(scalapack REQUIRED)
    target_link_libraries(scalapack INTERFACE gfortran::gfortran)
    pkg_install(cyclops)
    find_package(cyclops REQUIRED)
    target_link_libraries(tb-deps INTERFACE cyclops::cyclops)
    target_compile_definitions(tb-flags INTERFACE TB_MPI)
endif()

if(TB_ENABLE_MATX)
    find_package(cuTENSOR REQUIRED)
    find_package(cuTensorNet REQUIRED)
    pkg_install(matx)
    find_package(matx REQUIRED)
    target_link_libraries(tb-deps INTERFACE matx::matx)
endif()

# This fixes a nvcclink issue with libpthread.a being empty on ubuntu 22.04. It's enough to use the -fopenmp flag.
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_LINK_LIBRARIES "${OpenMP_CXX_FLAGS}")
