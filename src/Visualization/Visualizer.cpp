#include <ZividPython/Visualization/Visualizer.h>

#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/ReleasablePointCloud.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    class ReleasableFrame;

    void wrapClass(pybind11::class_<SingletonVisualizer> pyClass)
    {
        pyClass.def(py::init([] { return SingletonVisualizer{ []() { return Zivid::Visualization::Visualizer{}; } }; }))
            .def("show", py::overload_cast<>(&SingletonVisualizer::show))
            .def("hide", &SingletonVisualizer::hide)
            .def("run", &SingletonVisualizer::run)
            .def("close", &SingletonVisualizer::close)
            .def("resize", &SingletonVisualizer::resize, py::arg("h"), py::arg("w"))
            .def("reset_to_fit", &SingletonVisualizer::resetToFit)
            .def("show_full_screen", &SingletonVisualizer::showFullScreen)
            .def("show_maximized", &SingletonVisualizer::showMaximized)
            .def("set_window_title", &SingletonVisualizer::setWindowTitle, py::arg("title"))
            .def(
                "show",
                [](SingletonVisualizer &self, const ReleasablePointCloud &cloud) { return self.show(cloud.impl()); },
                py::arg("cloud"))
            .def(
                "show",
                [](SingletonVisualizer &self, const ReleasableUnorganizedPointCloud &cloud) {
                    return self.show(cloud.impl());
                },
                py::arg("cloud"))
            .def(
                "show",
                [](SingletonVisualizer &self, const ReleasableFrame &frame) { return self.show(frame.impl()); },
                py::arg("frame"))
            .def_property("colors_enabled", &SingletonVisualizer::colorsEnabled, &SingletonVisualizer::setColorsEnabled)
            .def_property(
                "meshing_enabled", &SingletonVisualizer::isMeshingEnabled, &SingletonVisualizer::setMeshingEnabled)
            .def_property(
                "axis_indicator_enabled",
                &SingletonVisualizer::isAxisIndicatorEnabled,
                &SingletonVisualizer::setAxisIndicatorEnabled);
    }
} // namespace ZividPython
