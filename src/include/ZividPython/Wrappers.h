#pragma once

#include <sstream>
#include <string>

#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ZividPython/Traits.h>
#include <ZividPython/Wrapper.h>

#include <pybind11/pybind11.h>

#include <string_view>

namespace ZividPython
{
    namespace
    {
        template<typename T>
        using bool_t = decltype(static_cast<bool>(std::declval<T>()));

        /// Inserts underscore on positive case flank and before negative case flank:
        /// ZividSDKVersion -> zivid_sdk_version
        std::string toSnakeCase(std::string_view upperCamelCase)
        {
            if(upperCamelCase.empty())
            {
                throw std::invalid_argument{ "String is empty." };
            }

            if(!isupper(upperCamelCase[0]))
            {
                std::stringstream msg;
                msg << "First character of string: '" << upperCamelCase << "' is not capitalized";
                throw std::invalid_argument{ msg.str() };
            }
            std::stringstream ss;
            ss << char(tolower(upperCamelCase[0]));

            for(size_t i = 1; i < upperCamelCase.size(); ++i)
            {
                if(isupper(upperCamelCase[i]))
                {
                    auto previous = i - 1;
                    auto next = i + 1;

                    if(!isupper(upperCamelCase[previous])
                       || (next < upperCamelCase.size() && !isupper(upperCamelCase[next])))
                    {
                        ss << "_";
                    }
                }
                ss << char(tolower(upperCamelCase[i]));
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
        auto pyClass = pybind11::class_<Source>{ dest, exposedName, tags... }
                           .def("to_string", &Source::toString)
                           .def("__repr__", &Source::toString);

        if constexpr(is_detected<bool_t, Source>::value)
        {
            pyClass.def("__bool__", &Source::operator bool);
        }

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

    template<typename Source, typename WrapFunction, typename Destination>
    void wrapEnum(const Destination &dest, const char *exposedName, const WrapFunction &wrapFunction)
    {
        auto pyEnum = pybind11::enum_<Source>{ dest, exposedName };
        wrapFunction(pyEnum);
    }

    template<typename WrapFunction>
    void wrapNamespaceAsSubmodule(pybind11::module &dest,
                                  const WrapFunction &wrapFunction,
                                  const char *nonLowercaseName)
    {
        const std::string name = toSnakeCase(nonLowercaseName);
        auto submodule = dest.def_submodule(name.c_str());
        wrapFunction(submodule);
    }
} // namespace ZividPython

#define ZIVID_PYTHON_WRAP_CLASS(dest, name)                                                                            \
    ZividPython::wrapClass<name, ZividPython::WrapType::normal>(dest,                                                  \
                                                                static_cast<void (*)(pybind11::class_<name>)>(         \
                                                                    ZividPython::wrapClass),                           \
                                                                #name)

#define ZIVID_PYTHON_WRAP_ENUM_CLASS_BASE_IMPL(dest, name, source, callback)                                           \
    ZividPython::wrapEnum<source>(dest, name, callback)

#define ZIVID_PYTHON_WRAP_ENUM_CLASS(dest, name)                                                                       \
    ZIVID_PYTHON_WRAP_ENUM_CLASS_BASE_IMPL(dest,                                                                       \
                                           #name,                                                                      \
                                           name,                                                                       \
                                           static_cast<void (*)(pybind11::enum_<name>)>(ZividPython::wrapEnum))

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

#define ZIVID_PYTHON_WRAP_ARRAY2D_BUFFER_AS_RELEASABLE(dest, nativetype)                                               \
    ZividPython::wrapClass<ZividPython::ReleasableArray2D<nativetype>, ZividPython::WrapType::releasable>(             \
        dest,                                                                                                          \
        static_cast<void (*)(pybind11::class_<ZividPython::ReleasableArray2D<nativetype>>)>(ZividPython::wrapClass),   \
        "Array2D" #nativetype,                                                                                         \
        pybind11::buffer_protocol())

#define ZIVID_PYTHON_WRAP_DATA_MODEL(dest, name) ZividPython::wrapDataModel(dest, name{})

#define ZIVID_PYTHON_WRAP_NAMESPACE_AS_SUBMODULE(dest, name)                                                           \
    ZividPython::wrapNamespaceAsSubmodule(dest, ZividPython::name::wrapAsSubmodule, #name)
