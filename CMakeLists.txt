cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(zivid LANGUAGES CXX)

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
   message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
   file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                  "${CMAKE_BINARY_DIR}/conan.cmake")
endif()

find_package(Python3 "${PYTHON_INTERPRETER_VERSION}" EXACT REQUIRED COMPONENTS Interpreter Development)
include(${CMAKE_BINARY_DIR}/conan.cmake)

conan_cmake_configure(
    REQUIRES
        eigen/3.3.9
        pybind11/2.9.1
    GENERATORS cmake_find_package
    OPTIONS eigen:MPL2_only=True)

conan_cmake_autodetect(CONAN_SETTINGS)

set(CONAN_INSTALL_FOLDER "${CMAKE_BINARY_DIR}/conan_install")

conan_cmake_install(
    PATH_OR_REFERENCE .
    REMOTE conancenter
    SETTINGS ${CONAN_SETTINGS}
    INSTALL_FOLDER ${CONAN_INSTALL_FOLDER}
    )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CONAN_INSTALL_FOLDER})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")
OPTION(ZIVID_SDK_VERSION "Zivid SDK version to link with" 0.0.0)

if(MSVC)
    add_compile_options(/bigobj)
endif()

find_package(Eigen3 3.3.9 MODULE REQUIRED)
find_package(pybind11 2.9.1 MODULE REQUIRED)
find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)

add_subdirectory(src)
