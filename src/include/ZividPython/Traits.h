#pragma once

#include <type_traits>

#if __has_include(<experimental/type_traits>)
#    include <experimental/type_traits>
#else
namespace Detail
{
    // Fallback implementation of is_detected for compilers lacking <experimental/type_traits>
    struct nonesuch
    {
        ~nonesuch() = delete;
        nonesuch(nonesuch const &) = delete;
        void operator=(nonesuch const &) = delete;
    };

    template<class Default, class AlwaysVoid, template<class...> class Op, class... Args>
    struct detector
    {
        using value_t = std::false_type;
        using type = Default;
    };

    template<class Default, template<class...> class Op, class... Args>
    struct detector<Default, std::void_t<Op<Args...>>, Op, Args...>
    {
        using value_t = std::true_type;
        using type = Op<Args...>;
    };
} // namespace Detail
#endif

namespace ZividPython
{
#if __has_include(<experimental/type_traits>)
    // Has the experimental header, so just use that.
    template<template<class...> class Op, class... Args>
    using is_detected = typename std::experimental::is_detected<Op, Args...>;
#else
    // Use fallback implementation.
    template<template<class...> class Op, class... Args>
    using is_detected = typename Detail::detector<Detail::nonesuch, void, Op, Args...>::value_t;
#endif
} // namespace ZividPython