#pragma once

#include <Zivid/Application.h>
#include <Zivid/CameraAddress.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableCamera.h>
#include <ZividPython/ReleasableComputeDevice.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class SingletonApplication : public Singleton<Zivid::Application>
    {
    public:
        using Singleton<Zivid::Application>::Singleton;

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_CONTAINER_RETURN(std::vector, ReleasableCamera, cameras)

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableCamera, connectCamera)

        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(
            ReleasableCamera,
            connectCamera,
            const Zivid::CameraInfo::SerialNumber &,
            serialNumber)

        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasableCamera, connectCamera, const Zivid::CameraAddress &, address)

        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasableCamera, createFileCamera, const std::string &, fileName)

        ReleasableCamera createFileCamera(const ReleasableFrame &frame)
        {
            return ReleasableCamera{ WITH_GIL_UNLOCKED(impl().createFileCamera(frame.impl())) };
        }

        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableComputeDevice, computeDevice)
    };

    void wrapClass(pybind11::class_<SingletonApplication> pyClass);
} // namespace ZividPython
