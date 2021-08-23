
if(TB_PACKAGE_MANAGER MATCHES "conan")
    # Find packages or install if missing
    find_package(Threads REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    find_package(Fortran REQUIRED)

    ##################################################################
    ### Installconanfile.txt dependencies                         ###
    ##################################################################

    if(TB_ENABLE_MKL)
        find_package(MKL COMPONENTS blas lapack gf gnu_thread lp64 REQUIRED)  # MKL - Intel's math Kernel Library, use the BLAS implementation in Eigen and Arpack. Includes lapack.
    endif()

    unset(CONAN_BUILD_INFO)
    unset(CONAN_BUILD_INFO CACHE)
    find_file(CONAN_BUILD_INFO
            conanbuildinfo.cmake
            HINTS ${CMAKE_BINARY_DIR} ${CMAKE_CURRENT_LIST_DIR}
            NO_DEFAULT_PATH)

    if(CONAN_BUILD_INFO)
        ##################################################################
        ### Use pre-existing conanbuildinfo.cmake                      ###
        ### This avoids having to run conan again                      ###
        ##################################################################
        message(STATUS "Detected Conan build info: ${CONAN_BUILD_INFO}")
        include(${CONAN_BUILD_INFO})
        conan_basic_setup(KEEP_RPATHS TARGETS)
    else()

        unset(CONAN_COMMAND CACHE)
        find_program (
                CONAN_COMMAND
                conan
                HINTS ${CONAN_PREFIX} $ENV{CONAN_PREFIX} ${CONDA_PREFIX} $ENV{CONDA_PREFIX}
                PATHS
                $ENV{HOME}/anaconda3
                $ENV{HOME}/miniconda3
                $ENV{HOME}/anaconda
                $ENV{HOME}/miniconda
                $ENV{HOME}/.local
                $ENV{HOME}/.conda
                PATH_SUFFIXES bin envs/tb/bin envs/dmrg/bin
        )
        if(NOT CONAN_COMMAND)
            message(FATAL_ERROR "Could not find conan program executable")
        else()
            message(STATUS "Found conan: ${CONAN_COMMAND}")
        endif()

        # Download cmake-conan automatically, you can also just copy the conan.cmake file
        if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
            message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
            file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.16.1/conan.cmake"
                    "${CMAKE_BINARY_DIR}/conan.cmake")
        endif()

        if(BUILD_SHARED_LIBS)
            list(APPEND TB_CONAN_OPTIONS OPTIONS "*:shared=True")
        else()
            list(APPEND TB_CONAN_OPTIONS OPTIONS "*:shared=False")
        endif()

        include(${CMAKE_BINARY_DIR}/conan.cmake)
        conan_cmake_run(
                CONANFILE conanfile.txt
                CONAN_COMMAND ${CONAN_COMMAND}
                BUILD_TYPE ${CMAKE_BUILD_TYPE}
                BASIC_SETUP CMAKE_TARGETS
                SETTINGS compiler.cppstd=17
                SETTINGS compiler.libcxx=libstdc++11
                PROFILE_AUTO ALL
                ${TB_CONAN_OPTIONS}
                KEEP_RPATHS
                BUILD missing
                BUILD openblas # This builds openblas everytime on github actions
        )
    endif()

    if(TARGET CONAN_PKG::eigen)
        target_compile_definitions(CONAN_PKG::eigen INTERFACE EIGEN_USE_THREADS)
    endif()

    if(TARGET CONAN_PKG::eigen)
        if(TB_EIGEN3_BLAS)
            if(TARGET mkl::mkl)
                message(STATUS "Eigen3 will use MKL")
                target_compile_definitions    (CONAN_PKG::eigen INTERFACE EIGEN_USE_MKL_ALL)
                target_compile_definitions    (CONAN_PKG::eigen INTERFACE EIGEN_USE_LAPACKE_STRICT)
                target_link_libraries         (CONAN_PKG::eigen INTERFACE mkl::mkl)
            elseif(TARGET CONAN_PKG::openblas)
                message(STATUS "Eigen3 will use OpenBLAS")
                target_compile_definitions    (CONAN_PKG::eigen INTERFACE EIGEN_USE_BLAS)
                target_compile_definitions    (CONAN_PKG::eigen INTERFACE EIGEN_USE_LAPACKE_STRICT)
                target_link_libraries         (CONAN_PKG::eigen INTERFACE CONAN_PKG::openblas)
            endif()
        endif()
    else()
        message(FATAL_ERROR "Target not defined: CONAN_PKG::eigen")
    endif()


    # Make aliases
    add_library(Eigen3::Eigen       ALIAS CONAN_PKG::eigen)
    add_library(h5pp::h5pp          ALIAS CONAN_PKG::h5pp)
    add_library(cxxopts::cxxopts    ALIAS CONAN_PKG::cxxopts)
    add_library(xtensor             ALIAS CONAN_PKG::xtensor)
    if(TARGET CONAN_PKG::openblas)
        add_library(OpenBLAS::OpenBLAS  ALIAS CONAN_PKG::openblas)
        #For convenience, define these targes
        add_library(BLAS::BLAS ALIAS CONAN_PKG::openblas)
        add_library(LAPACK::LAPACK ALIAS CONAN_PKG::openblas)
        add_library(lapacke::lapacke  ALIAS CONAN_PKG::openblas)
    endif()

endif()
