#pragma once

#include <Zivid/Experimental/Toolbox/PointCloudRegistration.h>
#include <ZividPython/Wrappers.h>

namespace ZividPython::Toolbox
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Toolbox

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::Experimental::Toolbox::LocalPointCloudRegistrationResult> pyClass);
} // namespace ZividPython
