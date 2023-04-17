function(find_hptt)
    if(NOT BUILD_SHARED_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    find_library(HPTT_LIBRARY
                 hptt
                 HINTS ${PKG_INSTALL_DIR}
                 PATH_SUFFIXES lib hptt/lib
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 QUIET
                 )
    find_path(HPTT_INCLUDE_DIR
              hptt.h
              HINTS ${TB_DEPS_INSTALL_DIR}
              PATH_SUFFIXES include hptt/include
              NO_CMAKE_ENVIRONMENT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              QUIET
              )
endfunction()

find_hptt()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hptt DEFAULT_MSG HPTT_LIBRARY HPTT_INCLUDE_DIR)

if (hptt_FOUND AND NOT TARGET hptt::hptt)
    add_library(hptt::hptt UNKNOWN IMPORTED)
    set_target_properties(hptt::hptt PROPERTIES IMPORTED_LOCATION "${HPTT_LIBRARY}")
    target_include_directories(hptt::hptt SYSTEM INTERFACE ${HPTT_INCLUDE_DIR})
endif()