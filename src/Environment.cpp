#include <Zivid/Environment.h>
#include <ZividPython/Environment.h>

#include <pybind11/pybind11.h>

namespace ZividPython::Environment
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        dest.def("data_path", &Zivid::Environment::dataPath);
    }
} // namespace ZividPython::Environment
