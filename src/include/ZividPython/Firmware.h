#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::Firmware
{
    MetaData wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Firmware
