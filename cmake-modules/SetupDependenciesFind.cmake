if(TB_DOWNLOAD_METHOD MATCHES "find")
    # Let cmake find our Find<package>.cmake modules
    list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_INSTALL_PREFIX})


    if(CMAKE_SIZEOF_VOID_P EQUAL 8 OR CMAKE_GENERATOR MATCHES "64")
        set(FIND_LIBRARY_USE_LIB64_PATHS ON)
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(FIND_LIBRARY_USE_LIB32_PATHS ON)
    endif()

    #set(CMAKE_FIND_DEBUG_MODE ON)

    if(TB_PREFER_CONDA_LIBS)
        list(APPEND CMAKE_PREFIX_PATH
                $ENV{CONDA_PREFIX}
                $ENV{HOME}/anaconda3/envs/tb
                $ENV{HOME}/anaconda/envs/tb
                $ENV{HOME}/miniconda3/envs/tb
                $ENV{HOME}/miniconda/envs/tb
                $ENV{HOME}/.conda/envs/tb
                $ENV{HOME}/anaconda3
                $ENV{HOME}/anaconda
                $ENV{HOME}/miniconda3
                $ENV{HOME}/miniconda
                $ENV{HOME}/.conda
                )
    endif()
    list(APPEND CMAKE_PREFIX_PATH
            $ENV{EBROOTIMKL}
            $ENV{EBROOTOPENBLAS}
            $ENV{EBROOTBLAS}
            $ENV{EBROOTLAPACK}
            $ENV{EBROOTARPACKMINNG}
            $ENV{EBROOTARPACKPLUSPLUS}
            $ENV{EBROOTGCC}
            $ENV{EBROOTOPENMPI}
            $ENV{EBROOTZLIB}
            $ENV{EBROOTGLOG}
            $ENV{EBROOTGFLAGS}
            $ENV{EBROOTCERES}
            $ENV{EBROOTSUITESPARSE}
            $ENV{EBROOTCXSPARSE}
            $ENV{EBROOTMETIS}
            $ENV{EBROOTCHOLMOD}
            $ENV{EBROOTCOLAMD}
            $ENV{EBROOTCCOLAMD}
            $ENV{EBROOTAMD}
            $ENV{EBROOTCAMD}
            $ENV{BLAS_DIR}
            $ENV{BLAS_ROOT}
            $ENV{ARPACKPP_DIR}
            )

endif()