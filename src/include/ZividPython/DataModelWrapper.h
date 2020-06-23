#pragma once

#include "DepdendentFalse.h"
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ZividPython/Wrappers.h"

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

        template<bool isRoot, typename Dest, typename Target>
        void wrapDataModel(Dest &dest, const Target &target)
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
                //pyClass.def(py::init<const std::string &>(), py::arg("file_name"))
                //    .def("save", &Target::save, py::arg("file_name"))
                //    .def("load", &Target::load, py::arg("file_name")); //TODO: fix når save/load bare tar et argument
                //pyClass.def("set_from_string",
                //py::overload_cast<const std::string &, const std::string &>(&Target::setFromString),
                //py::arg("path"),py::arg("string_value"));
            }

            // This is inside out because of bug in MSVC,
            // 'if constexpr' should really be inside the lambda
            if constexpr(Target::nodeType == Zivid::DataModel::NodeType::group)
            {
                pyClass.def("__bool__", [](const Target & /* value*/) { return true; }); // reconsider bool(iris)
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafValue)
            {
                //pyClass.def("set_from_string",
                //    py::overload_cast<const std::string &>(&Target::setFromString),
                //    py::arg("string_value"));
                pyClass.def("__bool__", [](const Target &value) {
                    return Target{ typename Target::ValueType{} } != value; // NOLINT
                });
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafDataModelList)
            {
                pyClass.def("__bool__", [](const Target &value) {
                    return !value.value().empty(); // NOLINT
                });
            }
            else
            {
                static_assert(DependentFalse<Target>::value, "Target NodeType is unsupported");
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

                    std::string name{ MemberType::name };
                    //std::transform(begin(name), end(name), begin(name), ::tolower);
                    name = toSnakeCase(name);

                    pyClass.def_property(
                        name.c_str(),
                        [](const Target &source) { return Detail::getHelper<MemberType>(source); },
                        [](Target &dest, const MemberType &value) { return dest.set(value); });
                });
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafValue)
            {
                using ValueType = typename Target::ValueType;

                if constexpr(std::is_enum_v<ValueType>)
                {
                    enum class foo
                    {
                    };
                    ZIVID_PYTHON_WRAP_ENUM_CLASS_BASE_IMPL(pyClass, "enum", ValueType, [](auto &pyEnum) {
                        for(const auto &value : Target::validValues())
                        {
                            pyEnum.value(Target{ value }.toString().c_str(), value);
                        }
                        pyEnum.export_values();
                    });
                }
                pyClass
                    .def(py::init<const ValueType &>(), py::arg("value"))

                    .def_property_readonly("value",
                                           [](const Target &target) -> std::optional<typename Target::ValueType> {
                                               if(hasValue(target))
                                               {
                                                   return target.value();
                                               }
                                               else
                                               {
                                                   return {};
                                               }
                                           });

                if constexpr(!std::is_same_v<ValueType, bool>)
                {
                    //pyClass.def(py::self > py::self); // NOLINT //TODO brightness feilet
                    //pyClass.def(py::self < py::self); // NOLINT
                }

                if constexpr(Zivid::DataModel::HasValidRange<Target>::value)
                {
                    pyClass.def_property_readonly("valid_range", [](const Target &target) {
                        const auto range = Target::validRange();
                        return std::make_pair(range.min(), range.max());
                    });
                }

                if constexpr(Zivid::DataModel::HasValidValues<Target>::value)
                {
                    pyClass.def_property_readonly("valid_values", [](const Target &target) {
                        return Target::validValues();
                        //return ValueType{};
                        //return std::vectorvalues;
                        //return std::vector<typename Target::Constraints>(values.at(0));
                        //return std::vector<typename Target::Constraints>();
                    });
                }

                if constexpr(Zivid::DataModel::HasValidSize<Target>::value)
                {
                    pyClass.def_property_readonly("valid_size", [](const Target &target) {
                        const auto size = Target::validSize();
                        return std::make_pair(size.min(), size.max());
                    });
                }
            }
            else if constexpr(Target::nodeType == Zivid::DataModel::NodeType::leafDataModelList)
            {
                using ValueType = typename Target::ValueType::value_type;

                //wrapDataModel<false>(pyClass, ValueType{});

                pyClass.def_property_readonly("value", &Target::value)
                    .def("append", [](Target &dest, ValueType value) { dest.emplaceBack(std::move(value)); })
                    .def("size", &Target::size)
                    .def("is_empty", &Target::isEmpty);
                // detail::list_accessor operator[](size_t index) const {return {*this, index};}
                // ???
                // Missing Settings::frames
                // access only wrap .at() (Frames.at(1234))
            }

            else
            {
                static_assert(DependentFalse<Target>::value, "Target NodeType is unsupported");
            }
        }
    } // namespace Detail

    template<typename Dest, typename Target>
    void wrapDataModel(Dest &dest, const Target &target)
    {
        Detail::wrapDataModel<true>(dest, target);
    }
} // namespace ZividPython
