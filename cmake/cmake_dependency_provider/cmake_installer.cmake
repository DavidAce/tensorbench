
unset(PKG_IS_RUNNING CACHE) # Remove from cache when this file is included

function(pkg_install_dependencies  package_name)
    if(NOT PKG_INSTALL_SUCCESS AND NOT PKG_IS_RUNNING)
        message(STATUS "pkg_install_dependencies running with package_name: ${package_name}")
        unset(PKG_INSTALL_SUCCESS CACHE)
        set(PKG_IS_RUNNING TRUE CACHE INTERNAL "" FORCE)
        include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PKGInstall.cmake)


        # Eigen3 numerical library (needed by ceres and h5pp)
        pkg_install(Eigen3)

        # cli11 for parsing cli arguments
        pkg_install(cli11)

        # h5pp for writing to file binary in format
        pkg_install(h5pp)

        # Backward for printing pretty stack traces
        pkg_install(Backward)


        if(TB_ENABLE_TBLIS)
            pkg_install(tblis)
        endif()
        if(TB_ENABLE_XTENSOR)
            pkg_install(xsimd)
            pkg_install(xtl)
            pkg_install(xtensor)
            pkg_install(xtensor-blas)
        endif()

        if(TB_ENABLE_CYCLOPS)
            pkg_install(hptt)
            pkg_install(scalapack)
            pkg_install(cyclops)
        endif()

        if(TB_ENABLE_MATX)
            pkg_install(matx)
        endif()



        set(PKG_INSTALL_SUCCESS TRUE CACHE BOOL "PKG dependency install has been invoked and was successful")
        set(PKG_IS_RUNNING FALSE CACHE INTERNAL "" FORCE)
    endif()

    find_package(${ARGN} BYPASS_PROVIDER)
endfunction()



function(pkg_install_dependencies_not_in_conan)
    if(NOT PKG_INSTALL_SUCCESS AND NOT PKG_IS_RUNNING)
        message(STATUS "pkg_install_dependencies running with package_name: ${package_name}")
        unset(PKG_INSTALL_SUCCESS CACHE)
        set(PKG_IS_RUNNING TRUE CACHE INTERNAL "" FORCE)
        include(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PKGInstall.cmake)


        if(TB_ENABLE_TBLIS)
            pkg_install(tblis)
        endif()
        if(TB_ENABLE_XTENSOR)
            pkg_install(xsimd)
            pkg_install(xtl)
            pkg_install(xtensor)
            pkg_install(xtensor-blas)
        endif()

        if(TB_ENABLE_CYCLOPS)
            find_package(MPI COMPONENTS CXX REQUIRED)
            pkg_install(hptt)
            pkg_install(cyclops)
        endif()





#
#
#
#
#
#
#
#        # For the PCG random number generator
#        pkg_install(pcg-cpp)
#
#        # Iterative Eigenvalue solver for a few eigenvalues/eigenvectors using Arnoldi method.
#        pkg_install(arpack-ng)
#
#        # C++ frontend for arpack-ng. Custom find module.
#        pkg_install(arpack++)
#
#        # Eigen3 numerical library (needed by ceres and h5pp)
#        pkg_install(Eigen3)
#
#        # cli11 for parsing cli arguments
#        pkg_install(cli11)
#
#        # h5pp for writing to file binary in format
#        pkg_install(h5pp)
#
#        # Backward for printing pretty stack traces
#        pkg_install(Backward)
#
#        # ceres-solver (for L-BFGS routine)
#        pkg_install(Ceres)
#
#        pkg_install(primme)
#
#        if(DMRG_ENABLE_TBLIS)
#            pkg_install(tblis)
#        endif()

#        target_link_libraries(dmrg-deps INTERFACE
#                CLI11::CLI11
#                pcg-cpp::pcg-cpp
#                h5pp::h5pp
#                arpack++::arpack++
#                Ceres::ceres
#                BLAS::BLAS
#                Backward::Backward
#                )

        set(PKG_INSTALL_SUCCESS TRUE CACHE BOOL "PKG dependency install has been invoked and was successful")
        set(PKG_IS_RUNNING FALSE CACHE INTERNAL "" FORCE)
    endif()

    find_package(${ARGN} BYPASS_PROVIDER)
endfunction()
