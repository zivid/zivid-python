#pragma once

#include <type_traits>

namespace ZividPython
{
    template<typename T>
    struct DependentFalse : std::false_type
    {};
} // namespace ZividPython
