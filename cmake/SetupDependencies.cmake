
#################################################################
### Preempt Threads::Threads                                   ###
### It's looked for in dependencies, so we make it right       ###
### before it's done wrong, i.e. with pthread instead of       ###
### -lpthread.                                                 ###
### Here we specify the linking twice                          ###
### 1) As string to make sure -lpthread gets sandwiched by     ###
###    -Wl,--whole-archive.... -Wl,--no-whole-archive          ###
###    -Wl,--whole-archive.... -Wl,--no-whole-archive          ###
### 2) As usual to make sure that if somebody links            ###
###    Threads::Threads, then any repeated pthread appended    ###
###    to the end (the wrong order causes linking errors)      ###
##################################################################
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(Threads::Threads INTERFACE rt dl)
find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(Fortran REQUIRED)

if (TB_ENABLE_MKL)
    if(TB_ENABLE_CYCLOPS)
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 scalapack blacs_openmpi REQUIRED)
    else()
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 REQUIRED)
    endif()

endif ()


include(cmake/SetupDependenciesCMake.cmake)
include(cmake/SetupDependenciesConan.cmake)

include(cmake/InstallPackage.cmake)
add_library(tb-deps INTERFACE)

# Acrotensor
if(TB_ENABLE_ACRO)
    install_package(acrotensor TARGET_NAME acrotensor::acrotensor_static
                    CMAKE_ARGS -DACROTENSOR_ENABLE_CUDA:BOOL=${TB_ENABLE_CUDA})
endif()

if(TB_ENABLE_TBLIS)
    install_package(tblis MODULE)
endif()
if(TB_ENABLE_XTENSOR AND TARGET TB_ENABLE_OPENBLAS)
    find_package(xtensor REQUIRED)
    install_package(xtensor-blas
                    CMAKE_ARGS -DUSE_OPENBLAS:BOOL=ON
                    TARGET_NAME xtensor-blas
                    DEPENDS xtl xsimd xtensor OpenBLAS::OpenBLAS)
    target_link_libraries(xtensor INTERFACE xtensor-blas)
endif()

if(TB_ENABLE_CYCLOPS)
    install_package(cyclops MODULE)
endif()


##################################################################
### Link all the things!                                       ###
##################################################################
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(tb-flags INTERFACE OpenMP::OpenMP_CXX)
else()
    target_compile_options(tb-flags INTERFACE -Wno-unknown-pragmas)
endif()

target_link_libraries(tb-deps INTERFACE h5pp::h5pp cxxopts::cxxopts)


# Configure Eigen
if(TARGET Eigen3::Eigen)
    target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_USE_THREADS)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})
    if(TARGET mkl::mkl)
        message(STATUS "Eigen3 will use MKL")
        target_compile_definitions    (Eigen3::Eigen INTERFACE EIGEN_USE_MKL_ALL)
        target_compile_definitions    (Eigen3::Eigen INTERFACE EIGEN_USE_LAPACKE_STRICT)
        target_link_libraries         (Eigen3::Eigen INTERFACE mkl::mkl)
    elseif(TARGET OpenBLAS::OpenBLAS)
        message(STATUS "Eigen3 will use OpenBLAS")
        target_compile_definitions    (Eigen3::Eigen INTERFACE EIGEN_USE_BLAS)
        target_compile_definitions    (Eigen3::Eigen INTERFACE EIGEN_USE_LAPACKE_STRICT)
        target_link_libraries         (Eigen3::Eigen INTERFACE OpenBLAS::OpenBLAS)
    endif()
else()
    message(FATAL_ERROR "Target not defined: Eigen3::Eigen")
endif()


if(TB_ENABLE_ACRO)
    target_link_libraries(tb-deps INTERFACE acrotensor::acrotensor_static)
endif()
if(TB_ENABLE_TBLIS)
    target_link_libraries(tb-deps INTERFACE tblis::tblis)
endif()
if(TB_ENABLE_CYCLOPS)
    target_link_libraries(tb-deps INTERFACE cyclops::cyclops)
endif()
if(TB_ENABLE_XTENSOR)
    target_link_libraries(tb-deps INTERFACE xtensor)
endif()

if(TB_ENABLE_MKL)
    target_compile_definitions(tb-deps INTERFACE TB_MKL)
endif()
if(TB_ENABLE_OPENBLAS)
    target_compile_definitions(tb-deps INTERFACE TB_OPENBLAS)
    target_link_libraries(tb-deps INTERFACE OpenBLAS::OpenBLAS)
endif()