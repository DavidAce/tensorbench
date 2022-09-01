function(find_tblis)
    unset(TBLIS_LIBRARY)
    unset(TBLIS_LIBRARY CACHE)
    find_library(TBLIS_LIBRARY
            tblis
            HINTS ${TB_DEPS_INSTALL_DIR}
            PATH_SUFFIXES lib tblis/lib
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
            )
    find_library(TCI_LIBRARY
                 tci
                 HINTS ${TB_DEPS_INSTALL_DIR}
                 PATH_SUFFIXES lib tblis/lib
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 )
    find_path(TBLIS_INCLUDE_DIR
            tblis/tblis.h
            HINTS ${TB_DEPS_INSTALL_DIR}
            PATH_SUFFIXES include tblis/include
            NO_CMAKE_ENVIRONMENT_PATH
            NO_SYSTEM_ENVIRONMENT_PATH
            NO_CMAKE_SYSTEM_PATH
            )
    find_path(TCI_INCLUDE_DIR
              tci/tci_global.h
              HINTS ${TB_DEPS_INSTALL_DIR}
              PATH_SUFFIXES include tci/include
              NO_CMAKE_ENVIRONMENT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              )
endfunction()

find_tblis()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(tblis
        DEFAULT_MSG
        TBLIS_LIBRARY TCI_LIBRARY TBLIS_INCLUDE_DIR TCI_INCLUDE_DIR)


if (tblis_FOUND)
    if(BUILD_SHARED_LIBS)
        set(TBLIS_LIBTYPE SHARED)
    else()
        set(TBLIS_LIBTYPE STATIC)
    endif()

    add_library(tblis::tblis ${TBLIS_LIBTYPE} IMPORTED)
    add_library(tblis::tci   ${TBLIS_LIBTYPE} IMPORTED)
    set_target_properties(tblis::tci   PROPERTIES IMPORTED_LOCATION "${TCI_LIBRARY}")
    set_target_properties(tblis::tblis PROPERTIES IMPORTED_LOCATION "${TBLIS_LIBRARY}")
    target_include_directories(tblis::tci SYSTEM INTERFACE ${TCI_INCLUDE_DIR})
    target_include_directories(tblis::tblis SYSTEM INTERFACE ${TBLIS_INCLUDE_DIR})
    target_link_libraries(tblis::tblis INTERFACE tblis::tci)
#    find_package(MPI COMPONENTS C REQUIRED)
    if(NOT BUILD_SHARED_LIBS)
        target_link_libraries(tblis::tblis INTERFACE atomic hwloc)
    endif()
endif()