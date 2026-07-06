#include <Zivid/CameraAddress.h>
#include <Zivid/CameraInfo.h>
#include <Zivid/ComputeDevice.h>
#include <Zivid/Detail/ToolchainDetector.h>

#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/SingletonApplication.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ZividPython
{
    void wrapClass(pybind11::class_<SingletonApplication> pyClass)
    {
        pyClass
            .def(py::init([] {
                // This method constructs a Zivid::Application and identifies the wrapper as the Zivid Python wrapper.
                // For users of the SDK: please do not use this method and construct the Zivid::Application directly instead.
                return SingletonApplication{ [](Zivid::Detail::EnvironmentInfo::Wrapper wrapper) {
                                                return Zivid::Detail::createApplicationForWrapper(wrapper);
                                            },
                                             Zivid::Detail::EnvironmentInfo::Wrapper::python };
            }))
            .def(
                py::init([](Zivid::CUDAContextPtr cudaContext) {
                    // Construct Application with user-provided CUDA context
                    return SingletonApplication{
                        [](Zivid::Detail::EnvironmentInfo::Wrapper wrapper,
                           Zivid::OpenCLContextPtr openclContext,
                           Zivid::CUDAContextPtr cudaCtx) {
                            return Zivid::Detail::createApplicationForWrapper(wrapper, openclContext, cudaCtx);
                        },
                        Zivid::Detail::EnvironmentInfo::Wrapper::python,
                        Zivid::OpenCLContextPtr{},
                        cudaContext
                    };
                }),
                py::arg("cuda_context"),
                "Construct Application with a user-provided CUDA context. "
                "Use this to ensure Zivid uses the same CUDA context as your application.")
            .def("cameras", &SingletonApplication::cameras)
            .def("connect_camera", [](SingletonApplication &application) { return application.connectCamera(); })
            .def(
                "connect_camera",
                [](SingletonApplication &application, const std::string &serialNumber) {
                    return application.connectCamera(Zivid::CameraInfo::SerialNumber{ serialNumber });
                },
                py::arg("serial_number"))
            .def(
                "connect_camera",
                [](SingletonApplication &application, const Zivid::CameraAddress &address) {
                    return application.connectCamera(address);
                },
                py::arg("address"))
            .def(
                "create_file_camera",
                py::overload_cast<const std::string &>(&SingletonApplication::createFileCamera),
                py::arg("frame_file"))
            .def(
                "create_file_camera",
                py::overload_cast<const ReleasableFrame &>(&SingletonApplication::createFileCamera),
                py::arg("frame"))
            .def("compute_device", &SingletonApplication::computeDevice);
    }
} // namespace ZividPython
