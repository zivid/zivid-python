#pragma once

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ZividPython/Wrapper.h>

namespace ZividPython
{
    struct MetaData
    {
        std::string doc;
    };

    enum class WrapType
    {
        normal,
        releasable,
        singleton
    };

    template<typename Source, WrapType wrapType, typename WrapFunction, typename... Tags>
    void wrapClass(const pybind11::module &dest,
                   const WrapFunction &wrapFunction,
                   const char *exposedName,
                   Tags... tags)
    {
        auto pyClass = pybind11::class_<Source>{ dest, exposedName, pybind11::dynamic_attr(), tags... }
                           .def("to_string", &Source::toString)
                           .def("__repr__", &Source::toString);

        if constexpr(WrapType::releasable == wrapType)
        {
            pyClass.def("release", &Source::release);
        }
        else if constexpr(WrapType::singleton == wrapType)
        {
            pyClass.def_static("release", &Source::release);
        }

        const auto metaData = wrapFunction(pyClass);

        pyClass.doc() = metaData.doc;
    }

    template<typename WrapFunction>
    void wrapNamespaceAsSubmodule(pybind11::module &dest,
                                  const WrapFunction &wrapFunction,
                                  const char *nonLowercaseName)
    {
        std::string name{ nonLowercaseName };
        std::transform(begin(name), end(name), begin(name), ::tolower);
        auto submodule = dest.def_submodule(name.c_str());

        const auto metaData = wrapFunction(submodule);

        submodule.doc() = metaData.doc;
    }
} // namespace ZividPython

#define ZIVID_PYTHON_WRAP_CLASS(dest, name)                                                                            \
    ZividPython::wrapClass<Zivid::name, ZividPython::WrapType::normal>(                                                \
        dest, static_cast<ZividPython::MetaData (*)(pybind11::class_<Zivid::name>)>(ZividPython::wrapClass), #name)

#define ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(dest, name)                                                              \
    ZividPython::wrapClass<ZividPython::Releasable##name, ZividPython::WrapType::releasable>(                          \
        dest,                                                                                                          \
        static_cast<ZividPython::MetaData (*)(pybind11::class_<ZividPython::Releasable##name>)>(                       \
            ZividPython::wrapClass),                                                                                   \
        #name)

#define ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(dest, name)                                                               \
    ZividPython::wrapClass<ZividPython::Singleton##name, ZividPython::WrapType::singleton>(                            \
        dest,                                                                                                          \
        static_cast<ZividPython::MetaData (*)(pybind11::class_<ZividPython::Singleton##name>)>(                        \
            ZividPython::wrapClass),                                                                                   \
        #name);

#define ZIVID_PYTHON_WRAP_CLASS_BUFFER(dest, name)                                                                     \
    ZividPython::wrapClass<Zivid::name, ZividPython::WrapType::normal>(                                                \
        dest,                                                                                                          \
        static_cast<ZividPython::MetaData (*)(pybind11::class_<Zivid::name>)>(ZividPython::wrapClass),                 \
        #name,                                                                                                         \
        pybind11::buffer_protocol())

#define ZIVID_PYTHON_WRAP_DATA_MODEL(dest, name) ZividPython::wrapDataModel(dest, Zivid::name{})

#define ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(dest, name)                                                           \
    ZividPython::wrapNamespaceAsSubmodule(dest, ZividPython::name::wrapAsSubmodule, #name)
