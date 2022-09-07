foreach(lang C CXX CUDA)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)

endforeach()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE STRING "") ### Write compile commands to file


set(CMAKE_CXX_FLAGS_INIT                 "-g -fno-strict-aliasing -fdiagnostics-color=always" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE_INIT         "-ffp-contract=fast" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG_INIT           "-fno-omit-frame-pointer -fstack-protector -D_FORTIFY_SOURCE=2" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT  "-fno-omit-frame-pointer -fstack-protector -D_FORTIFY_SOURCE=2" CACHE STRING "")


set(CMAKE_CUDA_FLAGS_INIT                 "-Xcompiler -fno-strict-aliasing -Xcompiler -fdiagnostics-color=always --expt-relaxed-constexpr -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored"
                                                        CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES              "75;80;86"    CACHE STRING "")
set(CMAKE_CUDA_SEPARABLE_COMPILATION      "OFF"         CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER              "g++-10"      CACHE STRING "")


set(CMAKE_FIND_LIBRARY_USE_LIB64_PATHS ON)
set(CMAKE_FIND_LIBRARY_USE_LIB32_PATHS ON)
