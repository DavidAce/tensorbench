
if (TB_PACKAGE_MANAGER STREQUAL "cmake")
    include(cmake/InstallPackage.cmake)

    # Set CMake build options
    if(OPENBLAS_TARGET)
        list(APPEND OpenBLAS_ARGS -DTARGET:STRING=${OPENBLAS_TARGET})
    endif()
    if(OPENBLAS_DYNAMIC_ARCH)
        list(APPEND OpenBLAS_ARGS -DDYNAMIC_ARCH:BOOL=${OPENBLAS_DYNAMIC_ARCH})
    endif()
    list(APPEND OpenBLAS_ARGS -DUSE_THREAD:BOOL=ON)
    list(APPEND OpenBLAS_ARGS -DBUILD_RELAPACK:BOOL=OFF)

    list(APPEND h5pp_ARGS -DEigen3_ROOT:PATH=${TB_DEPS_INSTALL_DIR})
    list(APPEND h5pp_ARGS -DH5PP_PACKAGE_MANAGER:STRING=cmake)
    list(APPEND h5pp_ARGS -DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE})

    list(APPEND glog_ARGS -Dgflags_ROOT:PATH=${TB_DEPS_INSTALL_DIR})

    list(APPEND Ceres_ARGS -DEigen3_ROOT:PATH=${TB_DEPS_INSTALL_DIR})
    list(APPEND Ceres_ARGS -Dgflags_ROOT:PATH=${TB_DEPS_INSTALL_DIR})
    list(APPEND Ceres_ARGS -Dglog_ROOT:PATH=${TB_DEPS_INSTALL_DIR})

    if (NOT BUILD_SHARED_LIBS)
        set(GFLAGS_COMPONENTS COMPONENTS)
        set(GFLAS_ITEMS nothreads_static)
    endif ()

    # Find packages or install if missing
    find_package(Threads REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    find_package(Fortran REQUIRED)

    if (TB_ENABLE_MKL)
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 REQUIRED)  # MKL - Intel's math Kernel Library, use the BLAS implementation in Eigen and Arpack. Includes lapack.
    endif ()

    if (TB_ENABLE_OPENBLAS AND NOT MKL_FOUND)
#        set(CMAKE_FIND_DEBUG_MODE ON)
        # If MKL is not on openblas will be used instead. Includes blas, lapack and lapacke
        install_package(OpenBLAS VERSION 0.3.17
                CMAKE_ARGS ${OpenBLAS_ARGS}
                DEPENDS gfortran::gfortran Threads::Threads)
        target_compile_definitions(OpenBLAS::OpenBLAS INTERFACE OPENBLAS_AVAILABLE)
        # Fix for OpenBLAS 0.3.9, which otherwise includes <complex> inside of an extern "C" scope.
        target_compile_definitions(OpenBLAS::OpenBLAS INTERFACE lapack_complex_float=std::complex<float>)
        target_compile_definitions(OpenBLAS::OpenBLAS INTERFACE lapack_complex_double=std::complex<double>)
        #For convenience, define these targes
        unset(BLAS_LIBRARIES CACHE)
        unset(LAPACK_LIBRARIES CACHE)
        get_target_property(BLAS_LIBRARIES OpenBLAS::OpenBLAS LOCATION)
        get_target_property(BLAS_openblas_LIBRARY OpenBLAS::OpenBLAS LOCATION)
        get_target_property(LAPACK_LIBRARIES OpenBLAS::OpenBLAS LOCATION)
        get_target_property(LAPACK_openblas_LIBRARY OpenBLAS::OpenBLAS LOCATION)
        set(BLA_VENDOR OpenBLAS)
        if(NOT BUILD_SHARED_LIBS)
            set(BLA_STATIC ON)
        endif()
        find_package(BLAS REQUIRED)
        find_package(LAPACK REQUIRED)
    endif ()

    # Eigen3 numerical library (needed by ceres and h5pp)
    install_package(Eigen3 VERSION 3.4 TARGET_NAME Eigen3::Eigen)
    # h5pp for writing to file binary in format
    install_package(h5pp VERSION 1.9.0 CMAKE_ARGS ${h5pp_ARGS})

    # cxxopts for parsing cli arguments
    install_package(cxxopts VERSION 2.2.0)


    if(TB_ENABLE_XTENSOR AND NOT TARGET xtensor)
        if(TB_ENABLE_OPENBLAS)
            set(XTENSOR_USE_OPENBLAS ON)
            find_package(OpenBLAS REQUIRED)
            find_package(BLAS REQUIRED)
            find_package(LAPACK REQUIRED)
            set(OpenBLAS_DIR ${CMAKE_INSTALL_PREFIX}/share/cmake/OpenBLAS)
        else()
            set(XTENSOR_USE_OPENBLAS OFF)
        endif()
        mark_as_advanced(XTENSOR_USE_OPENBLAS)
        list(APPEND xtensor_ARGS  "-Dxtl_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor_ARGS  "-Dxsimd_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor_ARGS  "-DXTENSOR_USE_OPENMP:BOOL=${TB_ENABLE_OPENMP}")
        list(APPEND xtensor-blas_ARGS  "-DUSE_OPENBLAS:BOOL=${XTENSOR_USE_OPENBLAS}")
        list(APPEND xtensor-blas_ARGS  "-Dxtensor_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor-blas_ARGS  "-Dxtl_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor-blas_ARGS  "-Dxsimd_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor-blas_ARGS  "-DOpenBLAS_ROOT:PATH=${CMAKE_INSTALL_PREFIX}")
        list(APPEND xtensor-blas_ARGS  "-DOpenBLAS_DIR:PATH=${CMAKE_INSTALL_PREFIX}/share/cmake/OpenBLAS")
        install_package(xsimd TARGET_NAME xsimd)
        install_package(xtl DEPENDS xsimd TARGET_NAME xtl)
        install_package(xtensor CMAKE_ARGS ${xtensor_ARGS} TARGET_NAME xtensor DEPENDS xtl xsimd)
        install_package(xtensor-blas CMAKE_ARGS ${xtensor-blas_ARGS} TARGET_NAME xtensor-blas DEPENDS xtl xsimd xtensor BLAS::BLAS)

        add_library(OpenMP::OpenMP_CXX_xtensor IMPORTED INTERFACE)
        target_compile_definitions(xtensor INTERFACE XTENSOR_USE_XSIMD)
        target_link_libraries(OpenMP::OpenMP_CXX_xtensor INTERFACE OpenMP::OpenMP_CXX)
        target_link_libraries(xtensor INTERFACE xsimd xtensor-blas)
    endif()




    # Configure Eigen

    if(TARGET Eigen3::Eigen)
        target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_USE_THREADS)
    endif()

    if(TARGET Eigen3::Eigen)
        get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
        target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${EIGEN3_INCLUDE_DIR})
        message(STATUS "TB_EIGEN3_BLAS ${TB_EIGEN3_BLAS}" )
        if(TB_EIGEN3_BLAS)
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
        endif()
        cmake_host_system_information(RESULT _host_name   QUERY HOSTNAME)
        if(_host_name MATCHES "tetralith|triolith")
            # AVX aligns 32 bytes (AVX512 aligns 64 bytes).
            # When running on Tetralith, with march=native, there can be alignment mismatch
            # in ceres which results in a segfault on free memory.
            # Something like "double free or corruption ..."
            #   * EIGEN_MAX_ALIGN_BYTES=16 works on Tetralith

            ### NOTE October 4 2020 ####
            #
            # Another flag that seems to fix weird release-only bugs is
            #       -fno-strict-aliasing

            ### NOTE August 26 2020 ####
            #
            # Ceres started crashing on Tetralith again using -march=native.
            # Tried to solve this issue once and for all.
            # I've tried the following flags during compilation of DMRG++ and ceres-solver:
            #
            #           -DEIGEN_MALLOC_ALREADY_ALIGNED=[none,0,1]
            #           -DEIGEN_MAX_ALIGN_BYTES=[none,16,32]
            #           -march=[none,native]
            #           -std=[none,c++17]
            #
            # Up until now, [0,16,none,none] has worked but now for some reason it stopped now.
            # I noticed the stdc++=17 flag was not being passed on conan builds, so ceres defaulted to -std=c++14 instead.
            # I fixed this in the conanfile.py of the ceres build. The -package-manager=cmake method already had this fixed.
            # When no Eigen flags were passed, and ceres-solver finally built with -std=c++17 the issues vanished.
            # In the end what worked was [none,none,native,c++17] in both DMRG++ and ceres-solver.
            # It is important that the same eigen setup is used in all compilation units, and c++17/c++14 seems to
            # make Eigen infer some of the flags differently. In any case, settinc c++17 and no flags for eigen anywhere
            # lets Eigen do its thing in the same way everywhere.

            #            message(STATUS "Applying special Eigen compile definitions for Tetralith: EIGEN_MAX_ALIGN_BYTES=16")
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MALLOC_ALREADY_ALIGNED=0) # May work to fix CERES segfault?
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MAX_ALIGN_BYTES=16)  # May work to fix CERES segfault?
        else()
            #            message(STATUS "Applying special Eigen compile definitions for general machines: EIGEN_MAX_ALIGN_BYTES=16")
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MALLOC_ALREADY_ALIGNED=0) # May work to fix CERES segfaults!!!
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MAX_ALIGN_BYTES=16)  # May work to fix CERES segfault?
        endif()
    else()
        message(FATAL_ERROR "Target not defined: Eigen3::Eigen")
    endif()

endif ()
