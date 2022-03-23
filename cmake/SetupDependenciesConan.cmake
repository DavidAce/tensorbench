
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

    # Download cmake-conan integrator
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan/conan.cmake")
        message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
        file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/release/0.17/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan/conan.cmake"
                EXPECTED_HASH MD5=52a255a933397fdce3d0937f9c737e98
                TLS_VERIFY ON)
    endif()
    include(${CMAKE_BINARY_DIR}/conan/conan.cmake)

    if(BUILD_SHARED_LIBS)
        list(APPEND TB_CONAN_OPTIONS OPTIONS "*:shared=True")
    else()
        list(APPEND TB_CONAN_OPTIONS OPTIONS "*:shared=False")
    endif()



    conan_add_remote(CONAN_COMMAND ${CONAN_COMMAND} NAME conan-dmrg URL https://thinkstation.duckdns.org/artifactory/api/conan/conan-dmrg)
    conan_cmake_autodetect(CONAN_AUTODETECT)
    conan_cmake_install(
            CONAN_COMMAND ${CONAN_COMMAND}
            BUILD missing outdated cascade
            GENERATOR cmake_find_package_multi
            SETTINGS ${CONAN_AUTODETECT}
            INSTALL_FOLDER ${CMAKE_BINARY_DIR}/conan
            ${TB_CONAN_OPTIONS}
            PATH_OR_REFERENCE ${CMAKE_SOURCE_DIR}
    )

    ##################################################################
    ### Find all the things!                                       ###
    ##################################################################
    if(NOT CONAN_CMAKE_SILENT_OUTPUT)
        set(CONAN_CMAKE_SILENT_OUTPUT OFF) # Default is off
    endif()
    list(PREPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/conan)
    list(PREPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR}/conan)
    # Use CONFIG to avoid MODULE mode. This is recommended for the cmake_find_package_multi generator

    find_package(Eigen3       3.4    REQUIRED CONFIG)
    find_package(h5pp         1.9.1  REQUIRED CONFIG)
    find_package(fmt          8.0.1  REQUIRED CONFIG)
    find_package(spdlog       1.9.2  REQUIRED CONFIG)
    find_package(xtensor      0.24.0 REQUIRED CONFIG)
    find_package(cxxopts      2.2.1  REQUIRED CONFIG)

    if(NOT TB_ENABLE_MKL)
        find_package(OpenBLAS 0.3.17 REQUIRED CONFIG)
        target_compile_definitions(OpenBLAS::OpenBLAS INTERFACE OPENBLAS_AVAILABLE)
        #For convenience, define these targes
        add_library(BLAS::BLAS ALIAS OpenBLAS::OpenBLAS)
        add_library(LAPACK::LAPACK ALIAS OpenBLAS::OpenBLAS)
        add_library(lapacke::lapacke  ALIAS OpenBLAS::OpenBLAS)
    endif()
endif()
