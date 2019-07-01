#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::HDR
{
    MetaData wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::HDR
