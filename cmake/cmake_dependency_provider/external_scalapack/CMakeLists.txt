cmake_minimum_required(VERSION 3.15)
project(external-scalapack)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)
ExternalProject_Add(external_scalapack
                    URL https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.0.tar.gz
                    URL_MD5 9ff561fb6c47d8649597d3b70b5ef0cb
                    BUILD_ALWAYS TRUE
                    PREFIX          ${CMAKE_BINARY_DIR}
                    INSTALL_DIR     ${CMAKE_INSTALL_PREFIX}
                    #Give Ninja access to the terminal.
                    USES_TERMINAL_DOWNLOAD TRUE
                    USES_TERMINAL_UPDATE TRUE
                    USES_TERMINAL_CONFIGURE TRUE
                    USES_TERMINAL_BUILD TRUE
                    USES_TERMINAL_INSTALL TRUE
                    USES_TERMINAL_TEST TRUE
                    UPDATE_COMMAND ""
                    CMAKE_ARGS
                    ${INIT_CACHE_ARG}
                    -DCMAKE_Fortran_FLAGS:STRING=-fallow-argument-mismatch
                    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
                    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                    -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
                    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
                    -DCMAKE_INSTALL_MESSAGE=LAZY
                    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
                    # scalapack options
                    -DUSE_OPTIMIZED_LAPACK_BLAS:BOOL=ON
                    -DSCALAPACK_BUILD_TESTS:BOOL=OFF
                    )