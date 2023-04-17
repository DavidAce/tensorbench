function(find_cyclops)
    if(NOT BUILD_SHARED_LIBS)
        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
    unset(CYCLOPS_LIBRARY)
    unset(CYCLOPS_LIBRARY CACHE)
    find_library(CYCLOPS_LIBRARY
                 ctf
                 HINTS ${TB_DEPS_INSTALL_DIR}
                 PATH_SUFFIXES lib ctf/lib cyclops/lib
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 QUIET
                 )
    find_path(CYCLOPS_INCLUDE_DIR
              ctf.hpp
              HINTS ${TB_DEPS_INSTALL_DIR}
              PATH_SUFFIXES include cyclops/include ctf/include
              NO_CMAKE_ENVIRONMENT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              QUIET
              )
endfunction()
#function(find_hptt)
#    if(NOT BUILD_SHARED_LIBS)
#        set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX})
#    endif()
#    find_library(HPTT_LIBRARY
#                 hptt
#                 HINTS ${TB_DEPS_INSTALL_DIR}
#                 PATH_SUFFIXES lib hptt/lib
#                 NO_CMAKE_ENVIRONMENT_PATH
#                 NO_SYSTEM_ENVIRONMENT_PATH
#                 NO_CMAKE_SYSTEM_PATH
#                 QUIET
#                 )
#    find_path(HPTT_INCLUDE_DIR
#              hptt.h
#              HINTS ${TB_DEPS_INSTALL_DIR}
#              PATH_SUFFIXES include hptt/include
#              NO_CMAKE_ENVIRONMENT_PATH
#              NO_SYSTEM_ENVIRONMENT_PATH
#              NO_CMAKE_SYSTEM_PATH
#              QUIET
#              )
#endfunction()

find_cyclops()
find_package(hptt REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cyclops DEFAULT_MSG CYCLOPS_LIBRARY CYCLOPS_INCLUDE_DIR)

if (cyclops_FOUND AND NOT TARGET cyclops::cyclops)
    add_library(cyclops::cyclops UNKNOWN IMPORTED)
    set_target_properties(cyclops::cyclops PROPERTIES IMPORTED_LOCATION "${CYCLOPS_LIBRARY}")
    target_include_directories(cyclops::cyclops SYSTEM INTERFACE ${CYCLOPS_INCLUDE_DIR})
    find_package(hptt)
    if(TARGET hptt::hptt)
        target_link_libraries(cyclops::cyclops INTERFACE hptt::hptt)
    endif()

    find_package(MPI COMPONENTS CXX REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    target_link_libraries(cyclops::cyclops INTERFACE BLAS::BLAS MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()