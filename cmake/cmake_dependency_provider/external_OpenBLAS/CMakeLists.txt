cmake_minimum_required(VERSION 3.15)
project(external-OpenBLAS)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)
if(NOT TARGET)
    set(TARGET GENERIC)
endif()
if(NOT DEFINED DYNAMIC_ARCH)
    set(DYNAMIC_ARCH OFF)
endif()


ExternalProject_Add(external_OpenBLAS
        URL      https://github.com/xianyi/OpenBLAS/archive/v0.3.17.tar.gz
        URL_MD5  5429954163bcbaccaa13e11fe30ca5b6
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
        ${INIT_CACHE_ARG}
        -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
        -DCMAKE_POLICY_DEFAULT_CMP0054=NEW
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
        #-DCMAKE_SKIP_BUILD_RPATH:BOOL=OFF
        #-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=OFF
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
        -DCMAKE_INSTALL_MESSAGE=LAZY
        -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
        # OpenBLAS flags
        -DTARGET:STRING=${TARGET}
        -DDYNAMIC_ARCH:BOOL=${DYNAMIC_ARCH}
        -DUSE_THREAD:BOOL=${USE_THREAD}
        -DBUILD_RELAPACK:BOOL=${BUILD_RELAPACK}
        -DBINARY=64
        )


if(CMAKE_BUILD_TYPE MATCHES "Debug")
    ExternalProject_Get_property(external_OpenBLAS INSTALL_DIR)
    set(LIBRARY_DIR ${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
    if(EXISTS ${LIBRARY_DIR}/libopenblas_d.so)
        ExternalProject_Add_Step(external_OpenBLAS
                add_symlink_to_debug_shared
                COMMENT "Add symlink to debug lib"
                DEPENDEES configure build install
                COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIBRARY_DIR}/libopenblas_d.so ${LIBRARY_DIR}/libopenblas.so
                )
    endif()
    if(EXISTS ${LIBRARY_DIR}/libopenblas_d.a)
        ExternalProject_Add_Step(external_OpenBLAS
                add_symlink_to_debug_shared
                COMMENT "Add symlink to debug lib"
                DEPENDEES configure build install
                COMMAND ${CMAKE_COMMAND} -E create_symlink ${LIBRARY_DIR}/libopenblas_d.a ${LIBRARY_DIR}/libopenblas.a
                )
    endif()

endif()