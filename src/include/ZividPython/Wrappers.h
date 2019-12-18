#pragma once

#include <sstream>
#include <string>

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ZividPython/Wrapper.h>

#include <pybind11/pybind11.h>

namespace ZividPython
{
    namespace
    {
        std::string toSnakeCase(const std::string upperCamelCase)
        {
            if(upperCamelCase.empty())
            {
                throw std::invalid_argument{ "String is empty." };
            }

            if(!isupper(upperCamelCase[0]))
            {
                throw std::runtime_error{ "First character of string: '" + upperCamelCase + "' is not capitalized" };
            }
            std::stringstream ss;
            ss << char(tolower(upperCamelCase[0]));

            for(auto i = 1; i < upperCamelCase.size(); ++i)
            {
                if(isupper(upperCamelCase[i]))
                {
                    auto previous = i - 1;
                    auto next = i + 1;

                    // if surrounded by capital case (looking at the B character): ABC -> abc
                    if(isupper(upperCamelCase[previous])
                       && ((next < upperCamelCase.size() && isupper(upperCamelCase[next]))
                           || (next >= upperCamelCase.size())))
                    {
                        ss << char(tolower(upperCamelCase[i]));
                    }
                    // all other cases results in adding an underscore first
                    else
                    {
                        ss << "_" << char(tolower(upperCamelCase[i]));
                    }
                }
                // already lower case stays lower case: a -> a
                else
                {
                    ss << char(upperCamelCase[i]);
                }
            }
            return ss.str();
        }
    } // namespace
    enum class WrapType
    {
        normal,
        releasable,
        singleton,
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
            pyClass.def("release", &Source::release).def("assert_not_released", &Source::assertNotReleased);
        }
        else if constexpr(WrapType::singleton == wrapType)
        {
            pyClass.def_static("release", &Source::release);
        }
        wrapFunction(pyClass);
    }

    template<typename Source, typename WrapFunction>
    void wrapEnum(const pybind11::module &dest, const WrapFunction &wrapFunction, const char *exposedName)
    {
        auto pyEnum = pybind11::enum_<Source>{ dest, exposedName, pybind11::dynamic_attr() };
        wrapFunction(pyEnum);
    }

    template<typename WrapFunction>
    void wrapNamespaceAsSubmodule(pybind11::module &dest,
                                  const WrapFunction &wrapFunction,
                                  const char *nonLowercaseName)
    {
        std::string name{ nonLowercaseName };
        auto submodule = dest.def_submodule(toSnakeCase(name).c_str());
        wrapFunction(submodule);
    }
} // namespace ZividPython

#define ZIVID_PYTHON_WRAP_CLASS(dest, name)                                                                            \
    ZividPython::wrapClass<name, ZividPython::WrapType::normal>(dest,                                                  \
                                                                static_cast<void (*)(pybind11::class_<name>)>(         \
                                                                    ZividPython::wrapClass),                           \
                                                                #name)

#define ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, name)                                                                       \
    ZividPython::wrapEnum<name>(dest, static_cast<void (*)(pybind11::enum_<name>)>(ZividPython::wrapEnum), #name)

#define ZIVID_PYTHON_WRAP_CLASS_AS_RELEASABLE(dest, name)                                                              \
    ZividPython::wrapClass<ZividPython::Releasable##name, ZividPython::WrapType::releasable>(                          \
        dest, static_cast<void (*)(pybind11::class_<ZividPython::Releasable##name>)>(ZividPython::wrapClass), #name)

#define ZIVID_PYTHON_WRAP_CLASS_AS_SINGLETON(dest, name)                                                               \
    ZividPython::wrapClass<ZividPython::Singleton##name, ZividPython::WrapType::singleton>(                            \
        dest, static_cast<void (*)(pybind11::class_<ZividPython::Singleton##name>)>(ZividPython::wrapClass), #name);

#define ZIVID_PYTHON_WRAP_CLASS_BUFFER_AS_RELEASABLE(dest, name)                                                       \
    ZividPython::wrapClass<ZividPython::Releasable##name, ZividPython::WrapType::releasable>(                          \
        dest,                                                                                                          \
        static_cast<void (*)(pybind11::class_<ZividPython::Releasable##name>)>(ZividPython::wrapClass),                \
        #name,                                                                                                         \
        pybind11::buffer_protocol())

#define ZIVID_PYTHON_WRAP_DATA_MODEL(dest, name) ZividPython::wrapDataModel(dest, name{})

#define ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(dest, name)                                                           \
    ZividPython::wrapNamespaceAsSubmodule(dest, ZividPython::name::wrapAsSubmodule, #name)
