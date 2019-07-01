#include <Zivid/Environment.h>
#include <ZividPython/Environment.h>

#include <pybind11/pybind11.h>

namespace ZividPython::Environment
{
    MetaData wrapAsSubmodule(pybind11::module &dest)
    {
        dest.def("data_path", &Zivid::Environment::dataPath);

        return { "Zivid environment, configured through environment variables" };
    }
} // namespace ZividPython::Environment
