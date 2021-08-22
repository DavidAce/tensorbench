cmake_minimum_required(VERSION 3.15)
project(external-acrotensor)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
option(ACROTENSOR_ENABLE_CUDA        ""  FALSE)

include(GNUInstallDirs)
include(ExternalProject)
ExternalProject_Add(external_acrotensor
        GIT_REPOSITORY https://github.com/DavidAce/acrotensor.git
        GIT_TAG 4754589aa5554e063eaea0b7cd1408698497df81
        GIT_PROGRESS TRUE
        PREFIX      ${CMAKE_BINARY_DIR}
        INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
        BUILD_ALWAYS TRUE
        #Give Ninja access to the terminal.
        USES_TERMINAL_DOWNLOAD TRUE
        USES_TERMINAL_UPDATE TRUE
        USES_TERMINAL_CONFIGURE TRUE
        USES_TERMINAL_BUILD TRUE
        USES_TERMINAL_INSTALL TRUE
        USES_TERMINAL_TEST TRUE

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
        # acrotensor flags
        -DACROTENSOR_ENABLE_CUDA:BOOL=${ACROTENSOR_ENABLE_CUDA}
        )
