#pragma once

#include <ZividPython/Traits.h>

#include <pybind11/pybind11.h>

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>

#define WITH_GIL_UNLOCKED(...)                                                                                         \
    [&, this] {                                                                                                        \
        pybind11::gil_scoped_release gilLock;                                                                          \
        return __VA_ARGS__;                                                                                            \
    }()

#define ZIVID_PYTHON_FORWARD_0_ARGS(functionName, ...)                                                                 \
    decltype(auto) functionName() __VA_ARGS__                                                                          \
    {                                                                                                                  \
        return WITH_GIL_UNLOCKED(impl().functionName());                                                               \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(returnType, functionName, ...)                                         \
    auto functionName() __VA_ARGS__                                                                                    \
    {                                                                                                                  \
        return returnType{ WITH_GIL_UNLOCKED(impl().functionName()) };                                                 \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_CONTAINER_RETURN(container, returnType, functionName, ...)                    \
    auto functionName() __VA_ARGS__                                                                                    \
    {                                                                                                                  \
        auto nativeContainer = WITH_GIL_UNLOCKED(impl().functionName());                                               \
        container<returnType> returnContainer;                                                                         \
        returnContainer.reserve(nativeContainer.size());                                                               \
        std::transform(                                                                                                \
            std::make_move_iterator(begin(nativeContainer)),                                                           \
            std::make_move_iterator(end(nativeContainer)),                                                             \
            std::back_inserter(returnContainer),                                                                       \
            [](auto &&nativeValue) { return returnType{ std::forward<decltype(nativeValue)>(nativeValue) }; });        \
        return returnContainer;                                                                                        \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS(functionName, arg1Type, arg1Name, ...)                                             \
    decltype(auto) functionName(arg1Type arg1Name) __VA_ARGS__                                                         \
    {                                                                                                                  \
        return WITH_GIL_UNLOCKED(impl().functionName(arg1Name));                                                       \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(returnType, functionName, arg1Type, arg1Name, ...)                     \
    auto functionName(arg1Type arg1Name) __VA_ARGS__                                                                   \
    {                                                                                                                  \
        return returnType{ WITH_GIL_UNLOCKED(impl().functionName(arg1Name)) };                                         \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS(functionName, arg1Type, arg1Name, arg2Type, arg2Name, ...)                         \
    decltype(auto) functionName(arg1Type arg1Name, arg2Type arg2Name) __VA_ARGS__                                      \
    {                                                                                                                  \
        return WITH_GIL_UNLOCKED(impl().functionName(arg1Name, arg2Name));                                             \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(returnType, functionName, arg1Type, arg1Name, arg2Type, arg2Name, ...) \
    auto functionName(arg1Type arg1Name, arg2Type arg2Name) __VA_ARGS__                                                \
    {                                                                                                                  \
        return returnType{ WITH_GIL_UNLOCKED(impl().functionName(arg1Name, arg2Name)) };                               \
    }

#define ZIVID_PYTHON_ADD_COMPARE(op)                                                                                   \
    template<typename Releaseable>                                                                                     \
    bool operator op(const Releaseable &other) const                                                                   \
    {                                                                                                                  \
        return WITH_GIL_UNLOCKED(impl() op other.impl());                                                              \
    }

#define ZIVID_PYTHON_ADD_COPY_CONSTRUCTOR(className)                                                                   \
    className(const className &other)                                                                                  \
        : Releasable{ other }                                                                                          \
    {}

namespace ZividPython
{
    template<typename T>
    class Releasable
    {
    public:
        template<typename... Args, std::enable_if_t<std::is_constructible_v<T, Args...>, int> = 0>
        explicit Releasable(Args &&...args)
            : m_impl{ std::make_optional<T>(std::forward<Args>(args)...) }
        {}

        decltype(auto) toString() const
        {
            return impl().toString();
        }

        auto &impl() const
        {
            if(!m_impl) throw std::runtime_error{ "Instance have been released" };
            return m_impl.value();
        }

        auto &impl()
        {
            if(!m_impl) throw std::runtime_error{ "Instance have been released" };
            return m_impl.value();
        }

        void release()
        {
            m_impl.reset();
        }

        // This function is required to verify that the buffer has not already
        // released in certain situations. Ideally it should not exist.
        void assertNotReleased()
        {
            std::ignore = impl();
        }

    private:
        std::optional<T> m_impl{ std::make_optional<T>() };
    };

    template<typename T>
    class Singleton
    {
    public:
        template<
            typename FactoryFunction,
            typename... Args,
            std::enable_if_t<std::is_invocable_r_v<T, FactoryFunction, Args...>, int> = 0>
        explicit Singleton(FactoryFunction &&create, Args &&...args)
        {
            // Keep the singleton alive forever to avoid races with
            // static variables that the singleton may need during destruction
            // This should be fixed a more elegant way!
            if(!globalImpl)
            {
                globalImpl = std::make_shared<T>(std::forward<FactoryFunction>(create)(std::forward<Args>(args)...));
            }
        }

        decltype(auto) toString() const
        {
            return impl().toString();
        }

        auto &impl() const
        {
            if(!globalImpl)
            {
                throw std::runtime_error{ "Instance has been released" };
            }
            return *globalImpl;
        }

        static void release()
        {
            globalImpl.reset();
        }

    private:
        static std::shared_ptr<T> globalImpl;
    };

    template<typename T>
    std::shared_ptr<T> Singleton<T>::globalImpl{ nullptr };
} // namespace ZividPython
