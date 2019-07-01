#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::Version
{
    MetaData wrapAsSubmodule(pybind11::module &dest);
} // namespace ZividPython::Version
