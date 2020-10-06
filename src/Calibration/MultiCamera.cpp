#include <ZividPython/Calibration/MultiCamera.h>
#include <ZividPython/Matrix.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Calibration::MultiCameraResidual> pyClass)
    {
        pyClass.def(py::init<float>(), py::arg("translation"))
            .def("translation", &Zivid::Calibration::MultiCameraResidual::translation);
    }
    void wrapClass(pybind11::class_<Zivid::Calibration::MultiCameraOutput> pyClass)
    {
        pyClass
            .def(py::init([](const std::vector<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> transforms,
                             const std::vector<Zivid::Calibration::MultiCameraResidual> residuals) {
                auto converted_transforms = std::vector<Zivid::Matrix<float, 4, 4>>();
                for(const auto &transform : transforms)
                {
                    converted_transforms.emplace_back(Conversion::toCpp(transform));
                }
                return std::make_unique<Zivid::Calibration::MultiCameraOutput>(converted_transforms, residuals);
            }))
            .def("valid", &Zivid::Calibration::MultiCameraOutput::valid)
            .def("transforms",
                 [](const Zivid::Calibration::MultiCameraOutput &calibrationOutputs) {
                     auto converted_transforms = std::vector<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>();
                     for(const auto &calibrationOutput : calibrationOutputs.transforms())
                     {
                         converted_transforms.emplace_back(Conversion::toPy(calibrationOutput));
                     }
                     return converted_transforms;
                 })
            .def("residuals", &Zivid::Calibration::MultiCameraOutput::residuals);
    }
} // namespace ZividPython
