#include <Zivid/DataModel/NodeType.h>

#include <ZividPython/DataModel.h>
#include <ZividPython/NodeType.h>
#include <ZividPython/Wrappers.h>

#include <pybind11/pybind11.h>

namespace ZividPython::DataModel
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        using namespace Zivid::DataModel;
        ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, NodeType);
    }
} // namespace ZividPython::DataModel