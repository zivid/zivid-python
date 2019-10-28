#pragma once

#include <ZividPython/Wrappers.h>

namespace ZividPython::HandEye
{
    void wrapClass(pybind11::class_<Zivid::HandEye::CheckerboardDetector> pyClass);
} // namespace ZividPython::HandEye
