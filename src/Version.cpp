#include <Zivid/Version.h>
#include <ZividPython/Version.h>

#include <pybind11/pybind11.h>

namespace ZividPython::Version
{
    MetaData wrapAsSubmodule(pybind11::module &dest)
    {
        dest.attr("major") = std::stoi(ZIVID_VERSION_MAJOR);
        dest.attr("minor") = std::stoi(ZIVID_VERSION_MINOR);
        dest.attr("patch") = std::stoi(ZIVID_VERSION_PATCH);
        dest.attr("build") = ZIVID_VERSION_BUILD;
        dest.attr("full") = ZIVID_VERSION;

        return { "Version information for the library" };
    }
} // namespace ZividPython::Version
