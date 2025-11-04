project(
    Visualization
    LANGUAGES
        CXX
        C
)

set(SOURCES
    Visualizer.cpp
    Wrapper.cpp
)

zivid_python_module_library(
    NAME
    ${PROJECT_NAME}
    SOURCES
    ${SOURCES}
    LINK_LIBRARIES
    Zivid::Visualization
    Eigen3::Eigen
    ZividPython::Core
)
