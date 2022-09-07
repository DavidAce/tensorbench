cmake_minimum_required(VERSION 3.15)

# Append search paths for find_package and find_library calls
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)


# Transform CMAKE_INSTALL_PREFIX to full path
if(DEFINED CMAKE_INSTALL_PREFIX
    AND NOT IS_ABSOLUTE CMAKE_INSTALL_PREFIX
    AND NOT CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    get_filename_component(CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
            ABSOLUTE BASE_DIR ${CMAKE_BINARY_DIR} CACHE FORCE)
    message(STATUS "Setting absolute path CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
endif()

# Setup build and install directories for dependencies
if(NOT TB_DEPS_BUILD_DIR)
    set(TB_DEPS_BUILD_DIR ${CMAKE_BINARY_DIR}/pkg-build)
endif()

# Install dependencies to the same location as the main project by default
if(NOT TB_DEPS_INSTALL_DIR)
    set(TB_DEPS_INSTALL_DIR ${CMAKE_INSTALL_PREFIX})
endif()


set(PKG_INSTALL_DIR ${TB_DEPS_INSTALL_DIR})
set(PKG_BUILD_DIR ${TB_DEPS_BUILD_DIR})

# Add search directories and flags for the CMake find_* tools
list(APPEND CMAKE_PREFIX_PATH $ENV{CMAKE_PREFIX_PATH} ${TB_DEPS_INSTALL_DIR} ${CMAKE_INSTALL_PREFIX})
list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE STRING "" FORCE)

# Make sure find_library prefers static/shared library depending on BUILD_SHARED_LIBS
# This is important when finding dependencies such as zlib which provides both shared and static libraries.
# Note that we do not force this cache variable, so users can override it
if(NOT BUILD_SHARED_LIBS)
    set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_STATIC_LIBRARY_SUFFIX};${CMAKE_SHARED_LIBRARY_SUFFIX}" CACHE INTERNAL "")
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8 OR CMAKE_GENERATOR MATCHES "64")
    set(FIND_LIBRARY_USE_LIB64_PATHS ON)
elseif (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(FIND_LIBRARY_USE_LIB32_PATHS ON)
endif ()


if(WIN32)
    # On Windows it is standard practice to collect binaries into one directory.
    # This way we avoid errors from .dll's not being found at runtime.
    # These directories will contain h5pp tests, examples and possibly dependencies
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin" CACHE PATH "Collect .exe and .dll")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Collect .lib")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib" CACHE PATH "Collect .lib")
endif()

if (TB_PACKAGE_MANAGER MATCHES "conan")
    # Paths to search for conan installation.
    list(APPEND TB_CONAN_CANDIDATE_PATHS
            ${CONAN_PREFIX}
            $ENV{CONAN_PREFIX}
            ${CONDA_PREFIX}
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
endif ()


mark_as_advanced(TB_CONAN_CANDIDATE_PATHS)