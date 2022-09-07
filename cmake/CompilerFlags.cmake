cmake_minimum_required(VERSION 3.20)
set(PROJECT_UNAME TB)
set(PROJECT_LNAME tb)

message(STATUS "C compiler ${CMAKE_C_COMPILER}")
message(STATUS "FC compiler ${CMAKE_Fortran_COMPILER}")
message(STATUS "CXX compiler ${CMAKE_CXX_COMPILER}")

############################################################
### Set  the same microarchitecture for c++ and OpenBLAS ###
############################################################

if(NOT TB_MICROARCH)
    set(TB_MICROARCH "native")
endif()
if(TB_MICROARCH)
    if (${TB_MICROARCH} STREQUAL "zen")
        string(TOUPPER ${TB_MICROARCH} OPENBLAS_MARCH)
        set(CXX_MARCH znver1)
    elseif (${TB_MICROARCH} STREQUAL "native")
        set(OPENBLAS_MARCH HASWELL)
        set(CXX_MARCH native)
    else()
        string(TOUPPER ${TB_MICROARCH} OPENBLAS_MARCH)
        string(TOLOWER ${TB_MICROARCH} CXX_MARCH)
    endif()
endif()


###  Add optional RELEASE/DEBUG compile to flags
if (NOT TARGET tb-flags)
    add_library(tb-flags INTERFACE)
endif ()

### Set arch
target_compile_options(tb-flags INTERFACE  $<$<COMPILE_LANGUAGE:CXX>:-march=${TB_MICROARCH}> )
target_compile_options(tb-flags INTERFACE  $<$<COMPILE_LANGUAGE:CXX>:-mtune=${TB_MICROARCH}> )
target_compile_options(tb-flags INTERFACE  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -march=${TB_MICROARCH}> )


###  Enable c++17 support
target_compile_features(tb-flags INTERFACE cxx_std_17)




# Settings for sanitizers
if (${PROJECT_UNAME}_ENABLE_ASAN)
    target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address;-fno-omit-frame-pointer>)
    target_link_options(tb-flags INTERFACE -fsanitize=address)
endif ()
if (${PROJECT_UNAME}_ENABLE_USAN)
    target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds;-fno-omit-frame-pointer>)
    target_link_libraries(tb-flags INTERFACE -fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds)
endif ()

### Enable link time optimization
function(target_enable_lto tgt)
    if(${PROJECT_UNAME}_ENABLE_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT lto_supported OUTPUT lto_error)
        if(lto_supported)
            message(STATUS "LTO enabled")
            set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ON)
        else()
            message(FATAL_ERROR "LTO is not supported: ${lto_error}")
        endif()
    endif()
endfunction()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND NOT CMAKE_EXE_LINKER_FLAGS MATCHES "fuse-ld=gold")
    set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=gold -Wl,--disable-new-dtags")
endif()

###############################
# Settings for shared builds
# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already (but later on when installing)
# Note: Since TB++ is often run from the build folder we want to keep the build-folder RPATH in the executable.
#       Therefore itt makes sense to keep this setting "FALSE" here but "TRUE" for dependencies that are
#       installed with in "fetch" mode with externalproject_add
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


