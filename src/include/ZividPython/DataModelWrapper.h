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

namespace
{
    template<typename T>
    class TypeIsZividRange : public std::false_type
    {};

    template<typename U>
    class TypeIsZividRange<Zivid::Range<U>> : public std::true_type
    {};

} // namespace

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
        struct TypeName<Zivid::PointXYZ>
        {
            static constexpr const char *value{ "(collections.abc.Iterable, _zivid.data_model.PointXYZ)" };
        };

        template<>
        struct TypeName<Zivid::Range<double>>
        {
            static constexpr const char *value{ "(collections.abc.Iterable, _zivid.data_model.Range)" };
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

        template<typename T>
        struct TypeName<T, std::enable_if_t<Zivid::DataModel::IsDataModelRoot<T>::value>>
        {
        private:
            class Helper
            {
                static constexpr auto concatenate()
                {
                    constexpr auto prefix{ "_zivid." };
                    constexpr size_t prefixLength{ std::char_traits<char>::length(prefix) };
                    constexpr size_t nameLength{ std::char_traits<char>::length(T::name) };

                    std::array<char, prefixLength + nameLength + 1> result{ '\0' };

                    // TODO: Switch to std::copy_n when we can switch to C++20 and drop support for GCC 9.
                    for(size_t i = 0; i < prefixLength; ++i)
                    {
                        result[i] = prefix[i];
                    }
                    for(size_t i = 0; i < nameLength; ++i)
                    {
                        result[prefixLength + i] = T::name[i];
                    }

                    return result;
                }

            public:
                static constexpr auto data{ concatenate() };
            };

        public:
            static constexpr const char *value = Helper::data.data();
        };

        template<typename DM, typename Dest, typename Node>
        void findAndWrapUninstantiatedNodesInDataModel(Dest &dest, const Node &node);

        template<bool isRoot, typename Dest, typename Target>
        py::class_<Target> wrapDataModel(Dest &dest, const Target &target, const bool uninstantiatedNode = false)
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

            pyClass.attr("uninstantiated_node") = uninstantiatedNode;

            if constexpr(isRoot)
            {
                pyClass.def(py::init<const std::string &>(), py::arg("file_name"))
                    .def("save", &Target::save, py::arg("file_name"))
                    .def("load", &Target::load, py::arg("file_name"))
                    .def_static("from_serialized", &Target::fromSerialized, py::arg("value"))
                    .def("serialize", &Target::serialize);
            }

            if constexpr(Target::nodeType == Zivid::DataModel::NodeType::group)
            {
                findAndWrapUninstantiatedNodesInDataModel<Target>(pyClass, target);

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
                            auto name = Target{ value }.toString();
                            if(name == "global")
                            {
                                // Special handling due to "global" being a reserved keyword in Python.
                                // The standard in these cases is to add a trailing underscore.
                                // https://peps.python.org/pep-0008/#naming-conventions
                                name = name + "_";
                            }
                            pyEnum.value(name.c_str(), value);
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

                pyClass.attr("is_array_like") = TypeIsZividRange<typename Target::ValueType>::value
                                                || std::is_same<typename Target::ValueType, Zivid::PointXYZ>::value;

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
                pyClass.attr("contained_type") = ValueTypeContained::name;

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

        constexpr bool isDirectChild(const std::string_view parent, const std::string_view child)
        {
            // Top-level nodes are always direct children of the root
            if(parent.empty() && child.find('/') == std::string::npos)
            {
                return true;
            }
            if(child.size() <= parent.size())
            {
                return false;
            }
            if(child.compare(0, parent.size(), parent) != 0)
            {
                return false;
            }
            // Ensure there are no additional path components
            const auto remainingPath = child.substr(parent.size());
            return remainingPath.find('/') == 0 && remainingPath.find('/', 1) == std::string::npos;
        }

        static_assert(isDirectChild("", "TopLevel") == true);
        static_assert(isDirectChild("", "TopLevel/WithChild") == false);
        static_assert(isDirectChild("TopLevel", "TopLevel/WithChild") == true);
        static_assert(isDirectChild("TopLevel", "TopLevel/WithChild/GrandChild") == false);

        template<typename DM, typename PyDest, typename Node>
        void findAndWrapUninstantiatedNodesInDataModel(PyDest &dest, const Node &node)
        {
            node.forEach([&](const auto &member) {
                using MemberType = std::remove_cv_t<std::remove_reference_t<decltype(member)>>;

                if constexpr(MemberType::nodeType == Zivid::DataModel::NodeType::group)
                {
                    findAndWrapUninstantiatedNodesInDataModel<DM>(dest, member);
                }
                else if constexpr(MemberType::nodeType == Zivid::DataModel::NodeType::leafDataModelList)
                {
                    using ValueTypeContained = typename MemberType::ValueType::value_type;

                    // Sanity check
                    static_assert(Zivid::DataModel::IsDataModelType<ValueTypeContained>::value);

                    if constexpr(Zivid::Detail::TypeTraits::IsInTuple<ValueTypeContained,
                                                                      typename DM::Descendants>::value)
                    {
                        // This node is instantiated.
                        return;
                    }

                    // Check via path that the contained type is a direct child of the dest
                    if constexpr(isDirectChild(DM::path, ValueTypeContained::path))
                    {
                        wrapDataModel<false>(dest, ValueTypeContained{}, true);
                    }
                }
            });
        }
    } // namespace Detail

    template<typename Dest, typename Target>
    py::class_<Target> wrapDataModel(Dest &dest, const Target &target)
    {
        return Detail::wrapDataModel<true>(dest, target);
    }
} // namespace ZividPython
