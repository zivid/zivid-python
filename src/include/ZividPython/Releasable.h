#pragma once

#include <optional>

#define ZIVID_PYTHON_FORWARD_0_ARGS(functionName)                                                                      \
    decltype(auto) functionName()                                                                                      \
    {                                                                                                                  \
        return impl().functionName();                                                                                  \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(ReturnType, functionName)                                              \
    auto functionName()                                                                                                \
    {                                                                                                                  \
        return ReturnType{ impl().functionName() };                                                                    \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_CONTAINER_RETURN(Container, ReturnType, functionName)                         \
    auto functionName()                                                                                                \
    {                                                                                                                  \
        auto nativeContainer = impl().functionName();                                                                  \
        Container<ReturnType> returnContainer;                                                                         \
        returnContainer.reserve(nativeContainer.size());                                                               \
        std::transform(std::make_move_iterator(begin(nativeContainer)),                                                \
                       std::make_move_iterator(end(nativeContainer)),                                                  \
                       std::back_inserter(returnContainer),                                                            \
                       [](auto &&nativeValue) {                                                                        \
                           return ReturnType{ std::forward<decltype(nativeValue)>(nativeValue) };                      \
                       });                                                                                             \
        return returnContainer;                                                                                        \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS(functionName, arg1type, arg1name)                                                  \
    decltype(auto) functionName(arg1type arg1name)                                                                     \
    {                                                                                                                  \
        return impl().functionName(arg1name);                                                                          \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(ReturnType, functionName, arg1type, arg1name)                          \
    auto functionName(arg1type arg1name)                                                                               \
    {                                                                                                                  \
        return ReturnType{ impl().functionName(arg1name) };                                                            \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS(functionName, arg1type, arg1name, arg2type, arg2name)                              \
    decltype(auto) functionName(arg1type arg1name, arg2type arg2name)                                                  \
    {                                                                                                                  \
        return impl().functionName(arg1name, arg2name);                                                                \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(ReturnType, functionName, arg1type, arg1name, arg2type, arg2name)      \
    auto functionName(arg1type arg1name, arg2type arg2name)                                                            \
    {                                                                                                                  \
        return ReturnType{ impl().functionName(arg1name, arg2name) };                                                  \
    }

#define ZIVID_PYTHON_ADD_COMPARE(op)                                                                                   \
    template<typename Releaseable>                                                                                     \
    bool operator op(const Releaseable &other) const                                                                   \
    {                                                                                                                  \
        return impl() op other.impl();                                                                                 \
    }

namespace ZividPython
{
    template<typename T>
    class Releasable
    {
    public:
        template<typename... Args>
        explicit Releasable(Args &&... args)
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

    private:
        std::optional<T> m_impl{ std::make_optional<T>() };
    };

    template<typename T>
    class Singleton
    {
    public:
        Singleton()
        {
            // Keep the singleton alive forever to avoid races with
            // static variables that the singleton may need during destruction
            // This should be fixed a more elegant way!
            if(!globalImpl) globalImpl = new T;
        }

        decltype(auto) toString() const
        {
            return impl().toString();
        }

        auto &impl() const
        {
            if(!globalImpl) throw std::runtime_error{ "Instance have been released" };
            return *globalImpl;
        }

        static void release()
        {
            delete globalImpl;
            globalImpl = nullptr;
        }

    private:
        static T *globalImpl;
    };

    template<typename T>
    T *Singleton<T>::globalImpl{ nullptr };
} // namespace ZividPython
