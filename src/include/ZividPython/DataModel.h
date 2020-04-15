#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::DataModel
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::DataModel
