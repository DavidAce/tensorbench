
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
    target_compile_definitions(mkl::mkl INTERFACE TB_MKL)
endif ()
if(TB_ENABLE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    target_compile_definitions(OpenBLAS::OpenBLAS INTERFACE TB_OPENBLAS)

endif()


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
if(TB_ENABLE_XTENSOR)
    find_package(xtensor REQUIRED)
    install_package(xtensor-blas
                    CMAKE_ARGS -DUSE_OPENBLAS:BOOL=ON
                    TARGET_NAME xtensor-blas
                    DEPENDS xtl xsimd xtensor BLAS::BLAS)
    target_compile_definitions(xtensor-blas INTERFACE HAVE_CBLAS=1)
    target_link_libraries(xtensor INTERFACE xtensor-blas)
endif()

if(TB_ENABLE_CYCLOPS)
    install_package(cyclops MODULE)
    target_compile_definitions(tb-flags INTERFACE TB_MPI)
endif()


##################################################################
### Link all the things!                                       ###
##################################################################
target_link_libraries(tb-flags INTERFACE OpenMP::OpenMP_CXX)
target_link_libraries(tb-deps INTERFACE
                      h5pp::h5pp
                      CLI11::CLI11
                      Backward::Backward)



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
    target_link_libraries(tb-deps INTERFACE mkl::mkl)
endif()
if(TB_ENABLE_OPENBLAS)
    target_link_libraries(tb-deps INTERFACE OpenBLAS::OpenBLAS)
endif()
target_compile_definitions(tb-flags INTERFACE EIGEN_USE_THREADS) # For Eigen::Tensor parallelization
set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_LINK_LIBRARIES "") # Use flag only
