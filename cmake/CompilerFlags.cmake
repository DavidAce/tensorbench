cmake_minimum_required(VERSION 3.20)
set(PROJECT_UNAME TB)

cmake_host_system_information(RESULT _host_name QUERY HOSTNAME)
if($ENV{CI} OR $ENV{GITHUB_ACTIONS})
    set(OPENBLAS_TARGET GENERIC CACHE INTERNAL "")
    set(OPENBLAS_DYNAMIC_ARCH ON CACHE INTERNAL "")
else()
    set(OPENBLAS_TARGET HASWELL CACHE INTERNAL "")
    set(OPENBLAS_DYNAMIC_ARCH ON CACHE INTERNAL "")
endif()

if (NOT TARGET tb-flags)
    add_library(tb-flags INTERFACE)
endif ()


###  Add optional RELEASE/DEBUG compile to flags
target_compile_options(tb-flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<CXX_COMPILER_ID:Clang>>: -fstandalone-debug>)
target_compile_options(tb-flags INTERFACE $<$<AND:$<CONFIG:RELWITHDEBINFO>,$<CXX_COMPILER_ID:Clang>>: -fstandalone-debug>)
target_compile_options(tb-flags INTERFACE
                       $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/W4>
                       $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-Wall -Wextra -Wpedantic -Wconversion -Wunused>
                       )


###  Enable c++17 support
target_compile_features(tb-flags INTERFACE cxx_std_17)

# Settings for sanitizers
if(COMPILER_ENABLE_ASAN)
    target_compile_options(tb-flags INTERFACE -fsanitize=address -fno-omit-frame-pointer)
    target_link_libraries(tb-flags INTERFACE -fsanitize=address)
endif()
if(COMPILER_ENABLE_USAN)
    target_compile_options(tb-flags INTERFACE -fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds -fno-omit-frame-pointer)
    target_link_libraries(tb-flags INTERFACE -fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds)
endif()

###  Link system libs statically
if(NOT BUILD_SHARED_LIBS)
    target_link_options(tb-flags INTERFACE -static-libgcc -static-libstdc++)
endif()

# Compiler-dependent linker flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_libraries(tb-flags INTERFACE -stdlib=libstdc++)
endif()

### Enable link time optimization
function(target_enable_lto tgt)
    if(CMAKE_EXE_LINKER_FLAGS MATCHES "mold")
        message(STATUS "LTO is not compatible with mold linker")
        return()
    endif()
    if(COMPILER_ENABLE_LTO)
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


# Try to use the mold linker (this disables LTO!)
find_program(MOLDLINKER NAMES mold ld.mold QUIET)
if(MOLDLINKER)
    message(STATUS "Found mold linker: [${MOLDLINKER}]")
    if(NOT CMAKE_EXE_LINKER_FLAGS MATCHES "-fuse-ld=")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold" CACHE INTERNAL "")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12.1.0)
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold" CACHE INTERNAL "")
            else()
                set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=mold" CACHE INTERNAL "")
                #            set(CMAKE_EXE_LINKER_FLAGS "-B${MOLDLINKER}" CACHE INTERNAL "")
            endif()
        endif()
    endif()
else()
    message(STATUS "Could not find linker [${linkername}] installed in the system. Using default.")
endif()

