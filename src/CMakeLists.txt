project(_zivid LANGUAGES CXX C)

set(SOURCES
    Calibration/Calibration.cpp
    Calibration/Detector.cpp
    Calibration/HandEye.cpp
    Calibration/MultiCamera.cpp
    Calibration/Pose.cpp
    CaptureAssistant.cpp
    DataModel.cpp
    Firmware.cpp
    InfieldCorrection/InfieldCorrection.cpp
    NodeType.cpp
    PixelMapping.cpp
    PointCloudExport.cpp
    Projection.cpp
    Presets.cpp
    ReleasableArray2D.cpp
    ReleasableCamera.cpp
    ReleasableFrame.cpp
    ReleasableFrame2D.cpp
    ReleasableImage.cpp
    ReleasablePointCloud.cpp
    ReleasableProjectedImage.cpp
    SingletonApplication.cpp
    Version.cpp
    Wrapper.cpp
    Matrix4x4.cpp
)

configure_file("Wrapper.h.in" "${CMAKE_CURRENT_BINARY_DIR}/include/ZividPython/Wrapper.h" @ONLY)

file(GLOB_RECURSE HEADERS
    "include/*.h"
    "${CMAKE_CURRENT_BINARY_DIR}/include/*h"
)

python3_add_library(
        ${PROJECT_NAME}
        MODULE
        WITH_SOABI
        ${SOURCES}
    )

target_include_directories(${PROJECT_NAME}
    PRIVATE include
            ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE Zivid::Core
            Python3::Module
            pybind11::pybind11
            Eigen3::Eigen
)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION modules/${PROJECT_NAME})
