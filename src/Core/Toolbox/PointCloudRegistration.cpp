#include <Zivid/Experimental/Toolbox/PointCloudRegistration.h>

#include <ZividPython/DataModelWrapper.h>
#include <ZividPython/Matrix.h>
#include <ZividPython/ReleasableUnorganizedPointCloud.h>
#include <ZividPython/Toolbox/PointCloudRegistration.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

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
