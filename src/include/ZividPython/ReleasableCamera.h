#pragma once

#include <Zivid/Camera.h>
#include <ZividPython/Releasable.h>
#include <ZividPython/ReleasableFrame.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython
{
    class ReleasableCamera : public Releasable<Zivid::Camera>
    {
    public:
        using Releasable<Zivid::Camera>::Releasable;

        ZIVID_PYTHON_ADD_COMPARE(==)
        ZIVID_PYTHON_ADD_COMPARE(!=)
        ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReleasableCamera, connect, const Zivid::Settings &, settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(disconnect)
        ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReleasableFrame, capture)
        ZIVID_PYTHON_FORWARD_0_ARGS(settings)
        ZIVID_PYTHON_FORWARD_1_ARGS(setSettings, const Zivid::Settings &, settings)
        ZIVID_PYTHON_FORWARD_0_ARGS(state)
        ZIVID_PYTHON_FORWARD_0_ARGS(modelName)
        ZIVID_PYTHON_FORWARD_0_ARGS(revision)
        ZIVID_PYTHON_FORWARD_0_ARGS(serialNumber)
        ZIVID_PYTHON_FORWARD_0_ARGS(firmwareVersion)
        ZIVID_PYTHON_FORWARD_0_ARGS(userDataMaxSizeBytes)
        ZIVID_PYTHON_FORWARD_1_ARGS(writeUserData, const std::vector<uint8_t> &, data)
        ZIVID_PYTHON_FORWARD_0_ARGS(userData)
    };

    MetaData wrapClass(pybind11::class_<ReleasableCamera> pyClass);
} // namespace ZividPython
