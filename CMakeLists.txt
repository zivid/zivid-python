cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(zivid LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

OPTION(ZIVID_PYTHON_VERSION "Version number to be compiled into the module" "UNKNOWN")
OPTION(ZIVID_SDK_VERSION "Zivid SDK version to link with" 0.0.0)

if(MSVC)
    add_compile_options(/bigobj)
endif()

find_package(Zivid ${ZIVID_SDK_VERSION} EXACT COMPONENTS Core REQUIRED)
find_package(pybind11 2.2.4 REQUIRED)

add_subdirectory(src)
