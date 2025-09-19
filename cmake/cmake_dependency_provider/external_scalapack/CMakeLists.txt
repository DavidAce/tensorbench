cmake_minimum_required(VERSION 3.15)
project(external-scalapack)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)
ExternalProject_Add(external_scalapack
                    GIT_REPOSITORY https://github.com/Reference-ScaLAPACK/scalapack.git
                    GIT_TAG 6423f17933eb9a2522814b78ab3c0d6da25ee85a
                    GIT_SHALLOW TRUE
                    GIT_PROGRESS TRUE
#                    URL https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.2.tar.gz
#                    URL_MD5 e3c3d8be031425663c14f1790dc87784
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