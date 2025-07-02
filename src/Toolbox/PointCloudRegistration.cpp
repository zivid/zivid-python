#include <Zivid/Experimental/Toolbox/PointCloudRegistration.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>
#include <ZividPython/Toolbox/PointCloudRegistration.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython::Toolbox
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::Experimental::Toolbox;

        ZIVID_PYTHON_WRAP_DATA_MODEL(dest, Zivid::Experimental::LocalPointCloudRegistrationParameters);
        ZIVID_PYTHON_WRAP_CLASS(dest, LocalPointCloudRegistrationResult);

        dest.def(
            "local_point_cloud_registration",
            [](const ZividPython::ReleasableUnorganizedPointCloud &target,
               const ZividPython::ReleasableUnorganizedPointCloud &source,
               const Zivid::Experimental::LocalPointCloudRegistrationParameters &param,
               const Zivid::Calibration::Pose &initialTransform) {
                return Zivid::Experimental::Toolbox::localPointCloudRegistration(
                    target.impl(), source.impl(), param, initialTransform);
            },
            py::arg("target"),
            py::arg("source"),
            py::arg("params"),
            py::arg("initial_transform"));
    }
} // namespace ZividPython::Toolbox

namespace ZividPython
{
    using namespace Zivid::Experimental::Toolbox;

    void wrapClass(pybind11::class_<LocalPointCloudRegistrationResult> pyClass)
    {
        pyClass.def("transform", &LocalPointCloudRegistrationResult::transform)
            .def("converged", &LocalPointCloudRegistrationResult::converged)
            .def("source_coverage", &LocalPointCloudRegistrationResult::sourceCoverage)
            .def("root_mean_square_error", &LocalPointCloudRegistrationResult::rootMeanSquareError);
    }
} // namespace ZividPython
