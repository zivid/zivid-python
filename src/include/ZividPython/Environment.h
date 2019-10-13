#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::Environment
{
    void wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Environment
