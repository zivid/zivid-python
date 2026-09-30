cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(TARGET Zivid::Core)
    message(STATUS "Building Python wrapper as part of Zivid SDK - using existing Zivid::Core target")
    set(ZIVID_PYTHON_SDK_BUILD TRUE)
else()
    message(STATUS "Building Python wrapper in standalone mode")
    set(ZIVID_PYTHON_SDK_BUILD FALSE)

    if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan_provider.cmake")
        message(STATUS "Downloading conan_provider.cmake from https://github.com/conan-io/cmake-conan")
        file(
            DOWNLOAD
                "https://raw.githubusercontent.com/conan-io/cmake-conan/develop2/conan_provider.cmake"
                "${CMAKE_BINARY_DIR}/conan_provider.cmake"
        )
    endif()
    set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES ${CMAKE_BINARY_DIR}/conan_provider.cmake)
endif()

project(ZividPython LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(ReadZividVersion)
include(CompilerWarningOptions)
include(RestoreStrippedIncludeDirs)
include(TargetUtilities)

if(NOT ZIVID_PYTHON_SDK_BUILD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

option(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")

if(NOT ZIVID_PYTHON_SDK_BUILD)
    zivid_read_sdk_version(JSON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/sdk_version.json")
    if("${ZIVID_SDK_VERSION}" STREQUAL "")
        message(FATAL_ERROR "ZIVID_SDK_VERSION is not set. Please ensure sdk_version.json is present and valid.")
    endif()
    message(STATUS "Standalone build - Using Zivid SDK version: ${ZIVID_SDK_VERSION}")
endif()

if(MSVC)
    add_compile_options(/bigobj)
endif()

if(NOT ZIVID_PYTHON_SDK_BUILD)
    find_package(
        Python3
        "${PYTHON_INTERPRETER_VERSION}"
        EXACT
        REQUIRED
        COMPONENTS
            Interpreter
            Development
    )
endif()
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

find_package(dlpack 1.2 CONFIG REQUIRED)
find_package(Eigen3 3.4.0 CONFIG REQUIRED)
find_package(pybind11 2.12.0 CONFIG REQUIRED)

if(NOT ZIVID_PYTHON_SDK_BUILD)
    find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)
    zivid_python_restore_stripped_include_dirs(
        Zivid::Core
        Zivid::Visualization
    )
endif()

add_subdirectory(src)

if(DEFINED ZIVID_PYTHON_OUTPUT_DIR AND ZIVID_PYTHON_OUTPUT_DIR)
    include(Zivid_PythonStage)

    zivid_stage_python_sources(
        TARGET ZividPythonStage_zivid
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/zivid"
        DEST_DIR "${ZIVID_PYTHON_OUTPUT_DIR}/zivid"
    )

    zivid_stage_python_sources(
        TARGET ZividPythonStage_zivid_init
        SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/_zivid"
        DEST_DIR "${ZIVID_PYTHON_OUTPUT_DIR}/_zivid"
    )

    add_dependencies(
        _zividcore
        ZividPythonStage_zivid
        ZividPythonStage_zivid_init
    )
    if(TARGET _zividvisualization)
        add_dependencies(
            _zividvisualization
            ZividPythonStage_zivid
            ZividPythonStage_zivid_init
        )
    endif()
endif()
