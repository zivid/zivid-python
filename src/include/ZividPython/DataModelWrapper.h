#pragma once

#include "ZividPython/Wrappers.h"
#include <ZividPython/DependentFalse.h>

#include <Zivid/DataModel/Traits.h>
#include <Zivid/Settings.h>
#include <Zivid/Settings2D.h>

#include <pybind11/chrono.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

namespace py = pybind11;

namespace ZividPython
{
    namespace Detail
    {
        // Need this indirection to work around an ICE in MSVC
        template<typename Target, typename Source>
        auto getHelper(const Source &s)
        {
            return s.template get<Target>();
        }

        template<typename T>
        bool hasValue(const T &leaf)
        {
            if constexpr(Zivid::DataModel::IsOptional<T>::value)
            {
                return leaf.hasValue();
            }
            return true;
        }

        template<typename T, typename = void>
        struct TypeName
        {
            static_assert(DependentFalse<T>::value, "Unexpected type");
        };

        template<typename T>
        struct TypeName<T, std::enable_if_t<std::is_enum_v<T>>>
        {
            static constexpr const char *value{ "enum" };
        };

        template<typename T>
        struct TypeName<std::vector<T>>
        {
            static constexpr const char *value{ "(collections.abc.Iterable,)" };
        };

        template<>
        struct TypeName<uint32_t>
        {
            static constexpr const char *value{ "(int,)" };
        };

        template<>
        struct TypeName<uint64_t>
        {
            static constexpr const char *value{ "(int,)" };
        };

        template<>
        struct TypeName<int>
        {
            static constexpr const char *value{ "(int,)" };
        };

        template<>
        struct TypeName<bool>
        {
            static constexpr const char *value{ "(bool,)" };
        };

        template<>
        struct TypeName<float>
        {
            static constexpr const char *value{ "(float, int,)" };
        };

        template<>
        struct TypeName<double>
        {
            static constexpr const char *value{ TypeName<float>::value };
        };

        template<>
        struct TypeName<std::chrono::microseconds>
        {
            static constexpr const char *value{ "(datetime.timedelta,)" };
        };

        template<>
        struct TypeName<std::chrono::milliseconds>
        {
            static constexpr const char *value{ "(datetime.timedelta,)" };
        };

        template<>
        struct TypeName<std::chrono::system_clock::time_point>
        {
            static constexpr const char *value{ "(datetime.datetime,)" };
        };

        template<>
        struct TypeName<std::string>
        {
            static constexpr const char *value{ "(str,)" };
        };

        template<typename T>
        struct TypeName<std::optional<T>>
        {
            static constexpr const char *value{ TypeName<T>::value };
        };

        template<bool isRoot, typename Dest, typename Target>
        py::class_<Target> wrapDataModel(Dest &dest, const Target &target)
        {
            py::class_<Target> pyClass{ dest, Target::name };

            pyClass.def(py::init())
                .def("__repr__", &Target::toString)
                .def("to_string", &Target::toString)
                .def(py::self == py::self) // NOLINT
                .def(py::self != py::self) // NOLINT
                .def_readonly_static("node_type", &Target::nodeType)
                .def_readonly_static("name", &Target::name)
                .def_readonly_static("path", &Target::path);

            if constexpr(isRoot)
            {
                pyClass.def(py::init<const std::string &>(), py::arg("file_name"))
                    .def("save", &Target::save, py::arg("file_name"))
                    .def("load", &Target::load, py::arg("file_name"));
            }

            if constexpr(Target::nodeType == Zivid::DataModel::NodeType::group)
            {
                // TODO: Workaround for no API to access uninstansiated nodes.
                // This generator should work on types and not instances.
                if constexpr(std::is_same_v<Target, Zivid::Settings> || std::is_same_v<Target, Zivid::Settings2D>)
                {
                    wrapDataModel<false>(pyClass, typename Target::Acquisition{});
                }

                target.forEach([&](const auto &member) {
                    wrapDataModel<false>(pyClass, member);

                    using MemberType = std::remove_const_t<std::remove_reference_t<decltype(member)>>;

                    const std::string name = toSnakeCase(MemberType::name);

                    pyClass.def_property(
                        name.c_str(),
                        [](const Target &read) { return Detail::getHelper<MemberType>(read); },
                        [](Target &write, const MemberType &value) { return write.set(value); });
                });
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafValue)
            {
                using ValueType = typename Target::ValueType;

                pyClass.attr("is_optional") = Zivid::DataModel::IsOptional<Target>::value;
                pyClass.attr("value_type") = TypeName<ValueType>::value;

                if constexpr(std::is_enum_v<ValueType>)
                {
                    ZIVID_PYTHON_WRAP_ENUM_CLASS_BASE_IMPL(pyClass, "enum", ValueType, [](auto &pyEnum) {
                        for(const auto &value : Target::validValues())
                        {
                            pyEnum.value(Target{ value }.toString().c_str(), value);
                        }
                        pyEnum.export_values();
                    });
                }

                if constexpr(!Zivid::DataModel::IsOptional<Target>::value)
                {
                    pyClass.def(py::init<const ValueType &>(), py::arg("value"));
                }
                else
                {
                    pyClass.def(py::init([](std::optional<ValueType> &value) {
                                    if(value)
                                    {
                                        return std::make_unique<Target>(value.value());
                                    }
                                    return std::make_unique<Target>();
                                }),
                                py::arg("value"));
                }

                pyClass.def_property_readonly("value",
                                              [](const Target &read) -> std::optional<typename Target::ValueType> {
                                                  if(hasValue(read))
                                                  {
                                                      return read.value();
                                                  }
                                                  else
                                                  {
                                                      return {};
                                                  }
                                              });

                if constexpr(Zivid::DataModel::HasValidRange<Target>::value)
                {
                    pyClass.def_property_readonly("valid_range", [](const Target &) {
                        const auto range = Target::validRange();
                        return std::make_pair(range.min(), range.max());
                    });
                    pyClass.def(py::self > py::self); // NOLINT
                    pyClass.def(py::self < py::self); // NOLINT
                }

                if constexpr(Zivid::DataModel::HasValidValues<Target>::value)
                {
                    pyClass.def_property_readonly("valid_values", [](const Target &) { return Target::validValues(); });
                }

                if constexpr(Zivid::DataModel::HasValidSize<Target>::value)
                {
                    pyClass.def_property_readonly("valid_size", [](const Target &) {
                        const auto size = Target::validSize();
                        return std::make_pair(size.min(), size.max());
                    });
                }
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafDataModelList)
            {
                using ValueTypeContainer = typename Target::ValueType;
                using ValueTypeContained = typename Target::ValueType::value_type;
                pyClass.attr("value_type") = TypeName<ValueTypeContainer>::value;
                pyClass.attr("is_optional") = Zivid::DataModel::IsOptional<Target>::value;

                pyClass.def_property_readonly("value", &Target::value)
                    .def("append", [](Target &write, ValueTypeContained value) { write.emplaceBack(std::move(value)); })
                    .def("size", &Target::size)
                    .def("is_empty", &Target::isEmpty)
                    .def("at", [](Target &write, const size_t index) { return write.at(index); });
            }
            else
            {
                static_assert(DependentFalse<Target>::value, "Target NodeType is unsupported");
            }
            return pyClass;
        }
    } // namespace Detail

    template<typename Dest, typename Target>
    py::class_<Target> wrapDataModel(Dest &dest, const Target &target)
    {
        return Detail::wrapDataModel<true>(dest, target);
    }
} // namespace ZividPython