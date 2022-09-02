function(find_cyclops)
    unset(CYCLOPS_LIBRARY)
    unset(CYCLOPS_LIBRARY CACHE)
    find_library(CYCLOPS_LIBRARY
                 ctf
                 HINTS ${TB_DEPS_INSTALL_DIR}
                 PATH_SUFFIXES lib ctf/lib cyclops/lib
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 )
    find_path(CYCLOPS_INCLUDE_DIR
              ctf.hpp
              HINTS ${TB_DEPS_INSTALL_DIR}
              PATH_SUFFIXES include cyclops/include ctf/include
              NO_CMAKE_ENVIRONMENT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              )
    message(STATUS "CYCLOPS_LIBRARY     : ${CYCLOPS_LIBRARY}")
    message(STATUS "CYCLOPS_INCLUDE_DIR : ${CYCLOPS_INCLUDE_DIR}")
endfunction()
function(find_hptt)
    find_library(HPTT_LIBRARY
                 hptt
                 HINTS ${TB_DEPS_INSTALL_DIR}
                 PATH_SUFFIXES lib hptt/lib
                 NO_CMAKE_ENVIRONMENT_PATH
                 NO_SYSTEM_ENVIRONMENT_PATH
                 NO_CMAKE_SYSTEM_PATH
                 )
    find_path(HPTT_INCLUDE_DIR
              hptt.h
              HINTS ${TB_DEPS_INSTALL_DIR}
              PATH_SUFFIXES include hptt/include
              NO_CMAKE_ENVIRONMENT_PATH
              NO_SYSTEM_ENVIRONMENT_PATH
              NO_CMAKE_SYSTEM_PATH
              )
endfunction()


find_cyclops()
find_hptt()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cyclops DEFAULT_MSG CYCLOPS_LIBRARY CYCLOPS_INCLUDE_DIR HPTT_LIBRARY HPTT_INCLUDE_DIR)

if (cyclops_FOUND)
    add_library(cyclops::cyclops UNKNOWN IMPORTED)
    set_target_properties(cyclops::cyclops PROPERTIES IMPORTED_LOCATION "${CYCLOPS_LIBRARY}")
    target_include_directories(cyclops::cyclops SYSTEM INTERFACE ${CYCLOPS_INCLUDE_DIR})

    add_library(hptt::hptt UNKNOWN IMPORTED)
    set_target_properties(hptt::hptt PROPERTIES IMPORTED_LOCATION "${HPTT_LIBRARY}")
    target_include_directories(hptt::hptt SYSTEM INTERFACE ${HPTT_INCLUDE_DIR})

    find_package(MPI COMPONENTS CXX REQUIRED)
    find_package(OpenMP COMPONENTS CXX REQUIRED)
    target_link_libraries(cyclops::cyclops INTERFACE hptt::hptt BLAS::BLAS MPI::MPI_CXX OpenMP::OpenMP_CXX)
endif()