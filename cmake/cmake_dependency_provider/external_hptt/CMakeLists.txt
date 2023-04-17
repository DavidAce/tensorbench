cmake_minimum_required(VERSION 3.15)
project(external-hptt)
if(INIT_CACHE_FILE)
    set(INIT_CACHE_ARG -C${INIT_CACHE_FILE})
endif()
include(GNUInstallDirs)
include(ExternalProject)
ExternalProject_Add(external_hptt
                    GIT_REPOSITORY https://github.com/springer13/hptt.git
                    GIT_TAG 942538649b51ff14403a0c73a35d9825eab2d7de # master on September 19, 2022
                    GIT_PROGRESS TRUE
                    PATCH_COMMAND patch  --forward -p1 -i ${PROJECT_SOURCE_DIR}/hptt.9425386.patch || true # Include omp and critter headers with <> instead of ""
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
                    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW
                    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                    -DCMAKE_INSTALL_RPATH:PATH=<INSTALL_DIR>/${CMAKE_INSTALL_LIBDIR}
                    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=ON
                    -DCMAKE_INSTALL_MESSAGE=LAZY
                    -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON
                    )