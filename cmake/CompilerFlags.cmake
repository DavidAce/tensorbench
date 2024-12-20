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
target_compile_options(tb-flags INTERFACE $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANG_AND_ID:CXX,Clang>>:-fstandalone-debug>)
target_compile_options(tb-flags INTERFACE $<$<AND:$<CONFIG:RELWITHDEBINFO>,$<COMPILE_LANG_AND_ID:C,Clang>>: -fstandalone-debug>)
target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4>
                                          $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic -Wconversion -Wunused>
                       )


###  Enable c++17 support
target_compile_features(tb-flags INTERFACE cxx_std_17)


# Settings for sanitizers
if(COMPILER_ENABLE_ASAN)
    target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address>) #-fno-omit-frame-pointer
    target_link_libraries(tb-flags INTERFACE -fsanitize=address)
endif()
if(COMPILER_ENABLE_USAN)
    target_compile_options(tb-flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds -fsanitize-undefined-trap-on-error>) #  -fno-omit-frame-pointer
    target_link_libraries(tb-flags INTERFACE -fsanitize=undefined,leak,pointer-compare,pointer-subtract,alignment,bounds -fsanitize-undefined-trap-on-error)
endif()

if(COMPILER_ENABLE_COVERAGE)
    target_compile_options(tb-flags INTERFACE --coverage)
    target_link_options(tb-flags INTERFACE --coverage)
endif()

# Enable static linking
function(target_enable_static_libgcc tgt)
    if(BUILD_SHARED_LIBS)
        return()
    endif()
    message(STATUS "Enabling static linking on target [${tgt}]")
    target_link_options(${tgt} BEFORE PUBLIC
                        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-static-libstdc++ -static-libgcc>
                        $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-static-libgcc>
                        )
endfunction()



