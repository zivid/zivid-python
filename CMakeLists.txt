cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan_provider.cmake")
    message(STATUS "Downloading conan_provider.cmake from https://github.com/conan-io/cmake-conan")
    file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/develop2/conan_provider.cmake"
            "${CMAKE_BINARY_DIR}/conan_provider.cmake")
endif()

set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES ${CMAKE_BINARY_DIR}/conan_provider.cmake)

project(zivid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")
OPTION(ZIVID_SDK_VERSION "Zivid SDK version to link with" 0.0.0)

if(MSVC)
    add_compile_options(/bigobj)
endif()

find_package(Python3 "${PYTHON_INTERPRETER_VERSION}" EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(Eigen3 3.4.0 CONFIG REQUIRED)
find_package(pybind11 2.12.0 CONFIG REQUIRED)
find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)

add_subdirectory(src)
