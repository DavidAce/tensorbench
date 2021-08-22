cmake_minimum_required(VERSION 3.15)
project(external-tblis)
include(../GetNumThreads.cmake)
get_num_threads(NUM_THREADS)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(ExternalProject)

#set(cflags "-O3 -std=c17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer")
#set(cxxflags "-O3 -std=c++17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer")

set(cflags "-O3 -std=c17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer")
set(cxxflags "-O3 -std=c++17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer")

#if(BUILD_SHARED_LIBS)
#    set(type solib)
#else()
#    set(type lib)
#endif()

ExternalProject_Add(external_tblis
        GIT_REPOSITORY  https://github.com/devinamatthews/tblis.git
        GIT_TAG  3845b1744c186ca3be45f9c9a69bcf527ac754b6
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
        CONFIGURE_COMMAND
            cd <SOURCE_DIR> &&
            pwd &&
            ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} --enable-threading=pthread --enable-config=haswell --prefix=<INSTALL_DIR>
#        ./configure CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} CFLAGS=${cflags} CXXFLAGS=${cxxflags} --enable-threading=pthread --enable-config=haswell --prefix=<INSTALL_DIR>

        BUILD_COMMAND
            cd <SOURCE_DIR> &&
            pwd &&
            make -j${NUM_THREADS}

        INSTALL_COMMAND
            cd <SOURCE_DIR> &&
            pwd &&
            make install
        )