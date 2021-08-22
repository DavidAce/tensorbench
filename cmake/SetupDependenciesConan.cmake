
if(TB_DOWNLOAD_METHOD MATCHES "conan")
    #  Make sure we use TB's own find modules
    list(INSERT CMAKE_MODULE_PATH 0  ${PROJECT_SOURCE_DIR}/cmake)
    ##############################################################################
    ###  Required OpenMP support                                               ###
    ###  Note that Clang has some  trouble with static openmp and that         ###
    ###  and that static openmp is not recommended. This tries to enable       ###
    ###  static openmp anyway because I find it useful. Installing             ###
    ###  libiomp5 might help for shared linking.                               ###
    ##############################################################################
    find_package(OpenMP REQUIRED) # Uses TB's own find module
    if(TARGET openmp::openmp)
        list(APPEND FOUND_TARGETS openmp::openmp)
    endif()

    ##################################################################
    ### Install conan-modules/conanfile.txt dependencies          ###
    ### This uses conan to get spdlog,eigen3,h5pp,ceres-solver    ###
    ###    h5pp/1.7.3@davidace/stable                             ###
    ###    eigen/3.3.7@davidace/patched                           ###
    ##################################################################

    if(TB_ENABLE_MKL)
        find_package(Fortran REQUIRED)
        include(cmake/SetupMKL.cmake)         # MKL - Intel's math Kernel Library, use the BLAS implementation in Eigen and Arpack. Includes lapack.
        if(TARGET mkl::mkl)
            list(APPEND FOUND_TARGETS mkl::mkl)
        endif()
    else()
        cmake_host_system_information(RESULT _host_name   QUERY HOSTNAME)
        if(${_host_name} MATCHES "travis|TRAVIS|Travis|fv-")
            message(STATUS "Setting dynamic arch for openblas")
            list(APPEND TB_CONAN_OPTIONS
                    OPTIONS openblas:dynamic_arch=True)
        endif()
        find_package(Fortran REQUIRED)
        list(APPEND FOUND_TARGETS gfortran::gfortran)
    endif()


    find_program (
            CONAN_COMMAND
            conan
            HINTS ${CONAN_PREFIX} $ENV{CONAN_PREFIX} ${CONDA_PREFIX} $ENV{CONDA_PREFIX}
            PATHS $ENV{HOME}/anaconda3  $ENV{HOME}/miniconda3 $ENV{HOME}/anaconda $ENV{HOME}/miniconda $ENV{HOME}/.conda
            PATH_SUFFIXES bin envs/tb/bin  envs/dmrg/bin
    )
    if(NOT CONAN_COMMAND)
        message(FATAL_ERROR "Could not find conan program executable")
    else()
        message(STATUS "Found conan: ${CONAN_COMMAND}")
    endif()

    # Download cmake-conan automatically, you can also just copy the conan.cmake file
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://github.com/conan-io/cmake-conan/raw/v0.15/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake")
    endif()

    include(${CMAKE_BINARY_DIR}/conan.cmake)
    conan_add_remote(NAME conan-center       URL https://conan.bintray.com)
    conan_add_remote(NAME conan-community    URL https://api.bintray.com/conan/conan-community/conan)
    conan_add_remote(NAME bincrafters        URL https://api.bintray.com/conan/bincrafters/public-conan)
    conan_add_remote(NAME conan-tb INDEX 1 URL https://api.bintray.com/conan/davidace/conan-tb)

    if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        # Let it autodetect libcxx
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        # There is no libcxx
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(APPEND TB_CONAN_SETTINGS SETTINGS compiler.libcxx=libstdc++11)
    endif()
    conan_cmake_run(
            CONANFILE conanfile.txt
            CONAN_COMMAND ${CONAN_COMMAND}
            BUILD_TYPE ${CMAKE_BUILD_TYPE}
            BASIC_SETUP CMAKE_TARGETS
            SETTINGS compiler.cppstd=17
            SETTINGS compiler.libcxx=libstdc++11
            ${TB_CONAN_SETTINGS}
            ${TB_CONAN_OPTIONS}
            BUILD missing
    )

    if(TARGET CONAN_PKG::Eigen3)
        set(eigen_target CONAN_PKG::Eigen3)
    elseif(TARGET CONAN_PKG::eigen)
        set(eigen_target CONAN_PKG::eigen)
    endif()


    if(TARGET ${eigen_target})
        if(TARGET openmp::openmp)
            target_compile_definitions    (${eigen_target} INTERFACE -DEIGEN_USE_THREADS)
        endif()
        if(TB_EIGEN3_BLAS)
            set(EIGEN3_USING_BLAS ON)
            if(TARGET mkl::mkl)
                message(STATUS "Eigen3 will use MKL")
                target_compile_definitions    (${eigen_target} INTERFACE -DEIGEN_USE_MKL_ALL)
                target_compile_definitions    (${eigen_target} INTERFACE -DEIGEN_USE_LAPACKE_STRICT)
                target_link_libraries         (${eigen_target} INTERFACE mkl::mkl)
            elseif(TARGET blas::blas)
                message(STATUS "Eigen3 will use OpenBLAS")
                target_compile_definitions    (${eigen_target} INTERFACE -DEIGEN_USE_BLAS)
                target_compile_definitions    (${eigen_target} INTERFACE -DEIGEN_USE_LAPACKE_STRICT)
                target_link_libraries         (${eigen_target} INTERFACE CONAN_PKG::openblas)
            endif()
        endif()

        # AVX2 aligns 32 bytes (AVX512 aligns 64 bytes).
        # When running on Tetralith, with march=native, there can be alignment mismatch
        # in ceres which results in a segfault on free memory.
        # Something like "double free or corruption ..."
        #   * EIGEN_MAX_ALIGN_BYTES=16 works on Tetralith
        cmake_host_system_information(RESULT _host_name  QUERY HOSTNAME)
        if(_host_name MATCHES "tetralith|triolith")
            message(STATUS "Applying special Eigen compile definitions for Tetralith: EIGEN_MAX_ALIGN_BYTES=32")
            target_compile_definitions(${eigen_target} INTERFACE EIGEN_MALLOC_ALREADY_ALIGNED=0) # May work to fix CERES segfaults!!!
            target_compile_definitions(${eigen_target} INTERFACE EIGEN_MAX_ALIGN_BYTES=32)
        else()
            #            message(STATUS "Applying special Eigen compile definitions for general machines")
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MALLOC_ALREADY_ALIGNED=1) # May work to fix CERES segfaults!!!
            #            target_compile_definitions(Eigen3::Eigen INTERFACE EIGEN_MAX_ALIGN_BYTES=32)
        endif()

    endif()

    if(TARGET CONAN::xtensor)
        add_library(xtensor ALIAS CONAN::xtensor)
    endif()


endif()
