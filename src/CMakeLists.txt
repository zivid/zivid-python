project(_zivid LANGUAGES CXX C)

set(SOURCES
    CameraRevision.cpp
    Environment.cpp
    Firmware.cpp
    HDR.cpp
    PointCloud.cpp
    ReleasableCamera.cpp
    ReleasableFrame.cpp
    SingletonApplication.cpp
    Version.cpp
    Wrapper.cpp
)

configure_file("Wrapper.h.in" "${CMAKE_CURRENT_BINARY_DIR}/include/ZividPython/Wrapper.h" @ONLY)

file(GLOB_RECURSE HEADERS
    "include/*.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/*h"
)

add_library(${PROJECT_NAME} MODULE ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME}
    PRIVATE include
            ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE Zivid::Core
            pybind11::module
)

set_target_properties(${PROJECT_NAME}
    PROPERTIES	
        PREFIX "${PYTHON_MODULE_PREFIX}"	
        SUFFIX "${PYTHON_MODULE_EXTENSION}"	
)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION modules/${PROJECT_NAME})
