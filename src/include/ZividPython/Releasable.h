#pragma once

#include <optional>
#include <stdexcept>

#define ZIVID_PYTHON_FORWARD_0_ARGS_TEMPLATE_1_ARG_WRAP_RETURN(returnType, functionName, returnTypeTypename)           \
    auto functionName()                                                                                                \
    {                                                                                                                  \
        return returnType{ impl().functionName<returnTypeTypename>() };                                                \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS(functionName)                                                                      \
    decltype(auto) functionName()                                                                                      \
    {                                                                                                                  \
        return impl().functionName();                                                                                  \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_TEMPLATE_1_ARG(functionName, returnTypeTypename)                                   \
    decltype(auto) functionName()                                                                                      \
    {                                                                                                                  \
        return impl().functionName<returnTypeTypename>();                                                              \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_RETURN(returnType, functionName)                                              \
    auto functionName()                                                                                                \
    {                                                                                                                  \
        return returnType{ impl().functionName() };                                                                    \
    }

#define ZIVID_PYTHON_FORWARD_0_ARGS_WRAP_CONTAINER_RETURN(container, returnType, functionName)                         \
    auto functionName()                                                                                                \
    {                                                                                                                  \
        auto nativeContainer = impl().functionName();                                                                  \
        container<returnType> returnContainer;                                                                         \
        returnContainer.reserve(nativeContainer.size());                                                               \
        std::transform(std::make_move_iterator(begin(nativeContainer)),                                                \
                       std::make_move_iterator(end(nativeContainer)),                                                  \
                       std::back_inserter(returnContainer),                                                            \
                       [](auto &&nativeValue) {                                                                        \
                           return returnType{ std::forward<decltype(nativeValue)>(nativeValue) };                      \
                       });                                                                                             \
        return returnContainer;                                                                                        \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS(functionName, arg1Type, arg1Name)                                                  \
    decltype(auto) functionName(arg1Type arg1Name)                                                                     \
    {                                                                                                                  \
        return impl().functionName(arg1Name);                                                                          \
    }

#define ZIVID_PYTHON_FORWARD_1_ARGS_WRAP_RETURN(returnType, functionName, arg1Type, arg1Name)                          \
    auto functionName(arg1Type arg1Name)                                                                               \
    {                                                                                                                  \
        return returnType{ impl().functionName(arg1Name) };                                                            \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS(functionName, arg1Type, arg1Name, arg2Type, arg2Name)                              \
    decltype(auto) functionName(arg1Type arg1Name, arg2Type arg2Name)                                                  \
    {                                                                                                                  \
        return impl().functionName(arg1Name, arg2Name);                                                                \
    }

#define ZIVID_PYTHON_FORWARD_2_ARGS_WRAP_RETURN(returnType, functionName, arg1Type, arg1Name, arg2Type, arg2Name)      \
    auto functionName(arg1Type arg1Name, arg2Type arg2Name)                                                            \
    {                                                                                                                  \
        return returnType{ impl().functionName(arg1Name, arg2Name) };                                                  \
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
