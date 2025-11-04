#include <Zivid/Version.h>
#include <ZividPython/Version.h>

#include <pybind11/pybind11.h>

namespace ZividPython::Version
{
    void wrapAsSubmodule(pybind11::module &dest)
    {
        dest.attr("major") = ZIVID_CORE_VERSION_MAJOR;
        dest.attr("minor") = ZIVID_CORE_VERSION_MINOR;
        dest.attr("patch") = ZIVID_CORE_VERSION_PATCH;
        dest.attr("build") = ZIVID_CORE_VERSION_BUILD;
        dest.attr("full") = ZIVID_CORE_VERSION;
    }
} // namespace ZividPython::Version
