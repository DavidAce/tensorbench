cmake_minimum_required(VERSION 3.15)
project(external-cyclops)
if (INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif ()
include(ExternalProject)
cmake_host_system_information(RESULT NUM_THREADS QUERY NUMBER_OF_PHYSICAL_CORES)

function(append_libs liblist addlibs)
    foreach(lib ${${addlibs}})
        if(${lib} MATCHES "^-" OR EXISTS ${lib})
            set(pfx "")
        else()
            set(pfx "-l")
        endif()
        list(APPEND applist ${pfx}${lib})
    endforeach()

    set(${liblist} "${${liblist}};${applist}" PARENT_SCOPE)
endfunction()

function(append_incs inclist addincs)
    message(inclist: ${inclist})
    message(addincs: ${addincs})
    foreach(inc ${${addincs}})
        list(APPEND applist -I${inc})
    endforeach()
    set(${inclist} "${${inclist}};${applist}" PARENT_SCOPE)
endfunction()

find_package(OpenMP COMPONENTS CXX REQUIRED)
find_package(MPI COMPONENTS CXX REQUIRED)
find_package(hptt REQUIRED)
find_package(BLAS REQUIRED)
find_package(scalapack REQUIRED)

if(TARGET scalapack AND NOT SCALAPACK_LIBRARY)
    get_target_property(SCALAPACK_LIBRARY scalapack LOCATION)
endif()

append_libs(libs BLAS_LIBRARIES)
append_libs(libs SCALAPACK_LIBRARY)
append_libs(libs HPTT_LIBRARY)
append_incs(incs HPTT_INCLUDE_DIR)

message(STATUS "Cyclops found OpenMP            : ${OpenMP_CXX_FLAGS}")
message(STATUS "Cyclops found BLAS_LIBRARIES    : ${BLAS_LIBRARIES}")
message(STATUS "Cyclops found SCALAPACK_LIBRARY : ${SCALAPACK_LIBRARY}")
message(STATUS "Cyclops found HPTT_LIBRARY      : ${HPTT_LIBRARY}")

append_libs(libs HPTT_LIBRARY)
string(REPLACE ";" " " libs "${libs}")
string(REPLACE ";" " " incs "${incs}")

set(cxx ${MPI_CXX_COMPILER})
set(ompi_cxx ${CMAKE_CXX_COMPILER})
set(cxxflags "-fPIC -std=c++17 ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
set(ldflags "${CMAKE_EXE_LINKER_FLAGS} -fPIC")

message(STATUS "cxx     : ${cxx}")
message(STATUS "ompi_cxx: ${ompi_cxx}")
message(STATUS "libs    : ${libs}")
message(STATUS "incs    : ${incs}")
message(STATUS "cxxflags: ${cxxflags}")
message(STATUS "ldflags : ${ldflags}")


ExternalProject_Add(external_cyclops
                    GIT_REPOSITORY https://github.com/cyclops-community/ctf.git
#                    GIT_TAG 8f1c19954613189528ee0243fad6c636f4b9033b # master on August 31, 2022
                    GIT_TAG c3f95829628c381dc9bf631c69f08a7b17580b53 # master on November 16, 2023
                    GIT_PROGRESS TRUE
                    PATCH_COMMAND patch  --forward -p1 -i ${PROJECT_SOURCE_DIR}/ctf.c3f9582.patch || true # Include omp and critter headers with <> instead of ""
                    PREFIX ${CMAKE_BINARY_DIR}
                    INSTALL_DIR ${CMAKE_INSTALL_PREFIX}

                    #Give Ninja access to the terminal.
                    USES_TERMINAL_DOWNLOAD TRUE
                    USES_TERMINAL_UPDATE TRUE
                    USES_TERMINAL_CONFIGURE TRUE
                    USES_TERMINAL_BUILD TRUE
                    USES_TERMINAL_INSTALL TRUE
                    USES_TERMINAL_TEST TRUE

                    BUILD_ALWAYS FALSE
                    BUILD_IN_SOURCE FALSE

                    CONFIGURE_COMMAND
                    COMMAND export CXX=${cxx}
                    COMMAND export OMPI_CXX=${ompi_cxx}
                    COMMAND export CXXFLAGS=${cxxflags}
                    COMMAND export LIBS=${libs}
                    COMMAND export LD_LIBS=${libs}
                    COMMAND export LD_FLAGS=${ldflags}
                    COMMAND export LDFLAGS=${ldflags}
                    COMMAND export LINKFLAGS=${ldflags}
                    COMMAND export INCLUDES=${incs}
                    COMMAND <SOURCE_DIR>/configure
                            --build-dir=<BINARY_DIR>
                            --install-dir=<INSTALL_DIR>
                            --with-lapack
                            --with-hptt
                            --verbose
                            ${build_scalapack}
                            ${with_scalapack}

                    BUILD_COMMAND
                    COMMAND export CXX=${cxx}
                    COMMAND export OMPI_CXX=${ompi_cxx}
                    COMMAND export CXXFLAGS=${cxxflags}
                    COMMAND export LIBS=${libs}
                    COMMAND export LD_LIBS=${libs}
                    COMMAND export LD_FLAGS=${ldflags}
                    COMMAND export LDFLAGS=${ldflags}
                    COMMAND export LINKFLAGS=${ldflags}
                    COMMAND export INCLUDES=${incs}
                    COMMAND make -j${NUM_THREADS}

                    INSTALL_COMMAND
                    COMMAND make install
#                    COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/hptt/include <INSTALL_DIR>/include
#                    COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/hptt/lib <INSTALL_DIR>/lib
                    )