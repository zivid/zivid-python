#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::Firmware
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Firmware
