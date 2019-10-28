#pragma once

#include <Zivid/HandEye/Pose.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapClass(pybind11::class_<Zivid::HandEye::Pose> pyClass);
} // namespace ZividPython
