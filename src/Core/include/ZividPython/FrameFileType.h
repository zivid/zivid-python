#pragma once

#include <Zivid/FrameFileType.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::FrameFileType> pyEnum);
} // namespace ZividPython
