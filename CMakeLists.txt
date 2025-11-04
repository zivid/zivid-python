cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan_provider.cmake")
    message(STATUS "Downloading conan_provider.cmake from https://github.com/conan-io/cmake-conan")
    file(
        DOWNLOAD
            "https://raw.githubusercontent.com/conan-io/cmake-conan/develop2/conan_provider.cmake"
            "${CMAKE_BINARY_DIR}/conan_provider.cmake"
    )
endif()

set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES ${CMAKE_BINARY_DIR}/conan_provider.cmake)

project(zivid LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(ReadZividVersion)
include(CompilerWarningOptions)
include(TargetUtilities)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")

zivid_read_sdk_version(JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/sdk_version.json")
if("${ZIVID_SDK_VERSION}" STREQUAL "")
    message(FATAL_ERROR "ZIVID_SDK_VERSION is not set. Please ensure sdk_version.json is present and valid.")
endif()

if(MSVC)
    add_compile_options(/bigobj)
endif()

find_package(
    Python3
    "${PYTHON_INTERPRETER_VERSION}"
    EXACT
    REQUIRED
    COMPONENTS
        Interpreter
        Development
)
if(WIN32)
    # If "debug" libraries for Python are installed on Windows, `find_package(Python3 ...)`
    # above will find debug version `_d` of imported python library during linking. `pybind11`
    # however expects to find release versions of imported python library during linking. This
    # leads to following linker error: `fatal error LNK1104: cannot open file 'python312.lib'`.
    # Hence we map imported python target properties to always use release libraries. See issue
    # https://github.com/pybind/pybind11/issues/3403 for details of the following workaround.
    set_target_properties(
        Python3::Module
        PROPERTIES
            MAP_IMPORTED_CONFIG_DEBUG
                ";RELEASE"
    )
endif()

find_package(Eigen3 3.4.0 CONFIG REQUIRED)
find_package(pybind11 2.12.0 CONFIG REQUIRED)
find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)

add_subdirectory(src)
