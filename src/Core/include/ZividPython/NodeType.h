#pragma once

#include <Zivid/DataModel/NodeType.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    void wrapEnum(pybind11::enum_<Zivid::DataModel::NodeType> pyEnum);
} // namespace ZividPython
