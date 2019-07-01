#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <algorithm>

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

        template<bool isRoot, typename Dest, typename Target>
        void wrapDataModel(Dest &dest, const Target &target)
        {
            pybind11::class_<Target> pyClass{ dest, Target::name, pybind11::dynamic_attr() };

            pyClass.def(pybind11::init<>())
                .def("__repr__", &Target::toString)
                .def("to_string", &Target::toString)
                .def("set_from_string", &Target::setFromString, pybind11::arg("string_value"))
                .def(pybind11::self == pybind11::self) // NOLINT
                .def(pybind11::self != pybind11::self) // NOLINT
                .def_readonly_static("is_container", &Target::isContainer)
                .def_readonly_static("name", &Target::name)
                .def_readonly_static("path", &Target::path)
                .doc() = Target::description;

            if constexpr(isRoot)
            {
                pyClass.def(pybind11::init<const std::string &>(), pybind11::arg("file_name"))
                    .def("save", &Target::save, pybind11::arg("file_name"))
                    .def("load", &Target::load, pybind11::arg("file_name"));
            }

            // This is inside out because of bug in MSVC,
            // 'if constexpr' should really be inside the lambda
            if constexpr(Target::isContainer)
            {
                pyClass.def("__bool__", [](const Target & /* value*/) { return true; });
            }
            else
            {
                pyClass.def("__bool__", [](const Target &value) {
                    return Target{ typename Target::ValueType{} } != value; // NOLINT
                });
            }

            if constexpr(!Target::isContainer)
            {
                pyClass.def(pybind11::init<const typename Target::ValueType &>(), pybind11::arg("value"))
                    .def_property_readonly("value", &Target::value);

                if constexpr(!std::is_same_v<typename Target::ValueType, bool>)
                {
                    pyClass.def(pybind11::self > pybind11::self); // NOLINT
                    pyClass.def(pybind11::self < pybind11::self); // NOLINT
                }

                if constexpr(!(std::is_same_v<typename Target::ValueType,
                                              bool> || std::is_same_v<typename Target::ValueType, std::string>))
                {
                    pyClass.def_property_readonly("range", [](const Target &t) {
                        const auto range = t.range();
                        return std::make_pair(range.min(), range.max());
                    });
                }
            }
            else
            {
                target.forEach([&](const auto &member) {
                    wrapDataModel<false>(pyClass, member);

                    using MemberType = std::remove_const_t<std::remove_reference_t<decltype(member)>>;

                    std::string name{ MemberType::name };
                    std::transform(begin(name), end(name), begin(name), ::tolower);

                    pyClass.def_property(
                        name.c_str(),
                        [](const Target &source) { return Detail::getHelper<MemberType>(source); },
                        pybind11::overload_cast<const MemberType &>(&Target::set));
                });
            }
        }
    } // namespace Detail

    template<typename Dest, typename Target>
    void wrapDataModel(Dest &dest, const Target &target)
    {
        Detail::wrapDataModel<true>(dest, target);
    }
} // namespace ZividPython
