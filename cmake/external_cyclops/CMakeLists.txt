cmake_minimum_required(VERSION 3.15)
project(external-cyclops)
include(../GetNumThreads.cmake)
get_num_threads(NUM_THREADS)
if (INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif ()
include(ExternalProject)

if(MKL_LIBRARIES)
    foreach(lib ${MKL_LIBRARIES})
        if(${lib} MATCHES "-Wl," OR EXISTS ${lib})
            set(pfx "")
        else()
            set(pfx "-l")
        endif()
        set(libs "${libs} ${pfx}${lib}")
    endforeach()
endif()
if(MKL_INCLUDE_DIRS)
    foreach(inc ${MKL_INCLUDE_DIRS})
        set(incs "${incs} -I${inc}")
    endforeach()
endif()

find_package(MPI COMPONENTS CXX REQUIRED)
message(STATUS "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")

find_package(OpenMP COMPONENTS CXX REQUIRED)
message(STATUS "OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")



set(cxx ${MPI_CXX_COMPILER})
set(ompi_cxx ${CMAKE_CXX_COMPILER})
set(cxxflags "-O3 -DNDEBUG -std=c++17 -fPIC -g -fno-strict-aliasing -fno-omit-frame-pointer ${OpenMP_CXX_FLAGS}")

message(STATUS "libs: ${libs}")


ExternalProject_Add(external_cyclops
                    GIT_REPOSITORY https://github.com/cyclops-community/ctf.git
                    GIT_TAG 8f1c19954613189528ee0243fad6c636f4b9033b # master on August 31, 2022
                    GIT_PROGRESS TRUE

                    PREFIX ${CMAKE_BINARY_DIR}
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

                    #Give Ninja access to the terminal.
                    USES_TERMINAL_DOWNLOAD TRUE
                    USES_TERMINAL_UPDATE TRUE
                    USES_TERMINAL_CONFIGURE TRUE
                    USES_TERMINAL_BUILD TRUE
                    USES_TERMINAL_INSTALL TRUE
                    USES_TERMINAL_TEST TRUE

                    BUILD_ALWAYS TRUE
                    BUILD_IN_SOURCE FALSE
                    CONFIGURE_COMMAND ""
                    BUILD_COMMAND
#                    COMMAND <SOURCE_DIR>/configure -h
                    COMMAND export CXX=${cxx}
                    COMMAND export OMPI_CXX=${ompi_cxx}
                    COMMAND export LIBS=${libs}
                    COMMAND export LD_LIBS=${ld_libs}
                    COMMAND export INCLUDES=${incs}
                    COMMAND export CXXFLAGS=${cxxflags}
                    COMMAND <SOURCE_DIR>/configure --build-dir=<BINARY_DIR> --install-dir=<INSTALL_DIR> --with-lapack --verbose

#                    BUILD_COMMAND
                    COMMAND make -j${NUM_THREADS}

                    INSTALL_COMMAND
                    make install
                    )