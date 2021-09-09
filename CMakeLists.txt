cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(zivid LANGUAGES CXX)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

find_package(Python3 "${PYTHON_INTERPRETER_VERSION}" EXACT REQUIRED COMPONENTS Interpreter Development.Module)
include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_run(
    REQUIRES
        eigen/3.3.9
        pybind11/2.6.1
    CMAKE_TARGETS
    BASIC_SETUP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")
OPTION(ZIVID_SDK_VERSION "Zivid SDK version to link with" 0.0.0)

if(MSVC)
    add_compile_options(/bigobj)
endif()

find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)

add_subdirectory(src)
