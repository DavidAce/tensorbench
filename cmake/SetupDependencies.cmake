
#################################################################
### Preempt Threads::Threads                                   ###
### It's looked for in dependencies, so we make it right       ###
### before it's done wrong, i.e. with pthread instead of       ###
### -lpthread.                                                 ###
### Here we specify the linking twice                          ###
### 1) As string to make sure -lpthread gets sandwiched by     ###
###    -Wl,--whole-archive.... -Wl,--no-whole-archive          ###
###    -Wl,--whole-archive.... -Wl,--no-whole-archive          ###
### 2) As usual to make sure that if somebody links            ###
###    Threads::Threads, then any repeated pthread appended    ###
###    to the end (the wrong order causes linking errors)      ###
##################################################################
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
target_link_libraries(Threads::Threads INTERFACE rt dl)


include(cmake/SetupDependenciesCMake.cmake)
include(cmake/SetupDependenciesConan.cmake)

include(cmake/InstallPackage.cmake)

# Acrotensor
if(TB_ENABLE_ACRO)
    list(APPEND acrotensor_ARGS  -DACROTENSOR_ENABLE_CUDA:BOOL=${TB_ENABLE_CUDA})
    install_package(acrotensor TARGET_NAME acrotensor::acrotensor_static CMAKE_ARGS ${acrotensor_ARGS})
endif()

if(TB_ENABLE_TBLIS)
    install_package(tblis MODULE)
endif()


##################################################################
### Link all the things!                                       ###
##################################################################
if(TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(tb-flags INTERFACE OpenMP::OpenMP_CXX)
else()
    target_compile_options(tb-flags INTERFACE -Wno-unknown-pragmas)
endif()

add_library(tb-deps INTERFACE)
target_link_libraries(tb-deps INTERFACE h5pp::h5pp ${CONAN_TARGETS})

if(TB_ENABLE_ACRO)
    target_link_libraries(tb-deps INTERFACE acrotensor::acrotensor_static)
endif()
if(TB_ENABLE_TBLIS)
    target_link_libraries(tb-deps INTERFACE tblis::tblis)
endif()
if(TB_ENABLE_XTENSOR)
    target_link_libraries(tb-deps INTERFACE xtensor)
endif()

if(TB_ENABLE_MKL)
    target_compile_definitions(tb-deps INTERFACE TB_MKL)
endif()
if(TB_ENABLE_OPENBLAS)
    target_compile_definitions(tb-deps INTERFACE TB_OPENBLAS)
    target_link_libraries(tb-deps INTERFACE OpenBLAS::OpenBLAS)
endif()