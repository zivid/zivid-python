option(WARNINGS "Enable compiler warnings" OFF)

if(WARNINGS)
    option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    if(WARNINGS)
        if(WARNINGS_AS_ERRORS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
        endif()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Wswitch-default -Weffc++")
        set(WARNINGS_THAT_SHOULD_BE_IGNORED # WHY it is ok to ignore
        )
        foreach(WARNING ${WARNINGS_THAT_SHOULD_BE_IGNORED})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-${WARNING}")
        endforeach()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    if(WARNINGS)
        if(WARNINGS_AS_ERRORS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")

        set(WARNINGS_THAT_SHOULD_BE_FIXED
            unsafe-buffer-usage # TODO(ZIVID-10837): Reenable this when we start using C++20
            gnu-zero-variadic-macro-arguments
        )

        set(WARNINGS_THAT_SHOULD_BE_IGNORED # WHY it is ok to ignore
            c++98-compat # Code base should be modern
            c++98-compat-pedantic # Code base should be modern
        )

        foreach(WARNING ${WARNINGS_THAT_SHOULD_BE_FIXED})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-${WARNING}")
        endforeach()

        foreach(WARNING ${WARNINGS_THAT_SHOULD_BE_IGNORED})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-${WARNING}")
        endforeach()
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
    endif()
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    if(WARNINGS)
        if(WARNINGS_AS_ERRORS)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
        endif()
        set(WARNINGS_THAT_SHOULD_BE_IGNORED # WHY it is ok to ignore
            4244 # Narrowing conversions: Too strict and noisy for this code base.
            4267 # Conversion: Happens a lot in this code, would complicate them too much to handle manually.
            4702 # Unreachable code: Ignoring because it triggers too many false positives on pybind11 wrapping code
        )
        foreach(WARNING ${WARNINGS_THAT_SHOULD_BE_IGNORED})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd${WARNING}")
        endforeach()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    endif()
else()
    message(WARNING "Unknown compiler, not able to set compiler options for ${CMAKE_CXX_COMPILER_ID}")
endif()
