#include <Zivid/Visualization/Visualizer.h>

#include <ZividPython/Wrappers.h>

#include <ZividPython/Visualization/Visualizer.h>
#include <ZividPython/VisualizationWrapper.h>

#include <pybind11/pybind11.h>

ZIVID_PYTHON_MODULE // NOLINT
{
    module.attr("__version__") = pybind11::str(ZIVID_PYTHON_VERSION);

    using namespace Zivid::Visualization;
    ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(module, Visualizer)
}
