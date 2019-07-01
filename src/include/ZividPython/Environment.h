#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::Environment
{
    MetaData wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Environment
