cmake_minimum_required(VERSION 3.15)
project(external-xtensor-blas)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)

option(USE_OPENBLAS "" OFF)
option(CXXBLAS_DEBUG "" OFF)

ExternalProject_Add(external_xtensor-blas
        GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
        GIT_TAG 0.19.1
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE

        PREFIX      ${CMAKE_BINARY_DIR}
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        #Give Ninja access to the terminal.
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_UPDATE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        USES_TERMINAL_TEST TRUE
        BUILD_ALWAYS TRUE

        CMAKE_ARGS
        # CMake flags
        ${INIT_CACHE_ARG}
        -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
        #-DCMAKE_SKIP_BUILD_RPATH:BOOL=OFF
        #-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=OFF
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
        -DCMAKE_INSTALL_MESSAGE=LAZY
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
        # xtensor-blas flags
#        -Dxtensor_ROOT:PATH=${xtensor_ROOT}
#        -Dxtl_ROOT:PATH=${xtl_ROOT}
#        -Dxsimd_ROOT:PATH=${xsimd_ROOT}
        -DUSE_OPENBLAS:BOOL=${USE_OPENBLAS}
        -DCXXBLAS_DEBUG:BOOL=${CXXBLAS_DEBUG}
        )
