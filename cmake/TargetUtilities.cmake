function(_zivid_python_library)
    set(OPTIONS "")
    set(ONE_VALUE_ARGUMENTS
        NAME
        TYPE
    )
    set(MULTI_VALUE_ARGUMENTS
        SOURCES
        LINK_LIBRARIES
    )
    cmake_parse_arguments("ARG" "" "${ONE_VALUE_ARGUMENTS}" "${MULTI_VALUE_ARGUMENTS}" ${ARGN})

    string(TOLOWER ${ARG_NAME} NAME_TOLOWER)
    set(MODULE_NAME "_zivid${NAME_TOLOWER}")

    if(${ARG_TYPE} STREQUAL "MODULE")
        set(ABI_OPTION "WITH_SOABI")
    endif()

    set(TEMPLATES_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../src/templates")

    # gersemi: off
    configure_file( # NOLINT
        "${TEMPLATES_DIR}/Wrapper.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/include/ZividPython/${ARG_NAME}Wrapper.h"
        @ONLY
    )
    # gersemi: on

    pybind11_add_module(
            ${MODULE_NAME}
            ${ARG_TYPE}
            ${ABI_OPTION}
            ${ARG_SOURCES}
            # NO_EXTRAS: prevents pybind11 from adding certain linker options that caused issues with some
            # compilers
            NO_EXTRAS
    )

    add_library("ZividPython::${ARG_NAME}" ALIAS ${MODULE_NAME})

    target_link_libraries(${MODULE_NAME} PRIVATE ${ARG_LINK_LIBRARIES})

    # Workaround for:
    # https://gitlab.kitware.com/cmake/cmake/-/issues/21676.
    set_target_properties(
        ${MODULE_NAME}
        PROPERTIES
            DEBUG_POSTFIX
                ""
    )

    target_include_directories(
        ${MODULE_NAME}
        PUBLIC
            include
            ${CMAKE_CURRENT_BINARY_DIR}/include/
    )

    # gersemi: off
    install( # NOLINT
        TARGETS ${MODULE_NAME}
        LIBRARY DESTINATION modules/_zivid
        RUNTIME DESTINATION modules/_zivid
    )
    # gersemi: on
endfunction()

function(zivid_python_shared_library)
    _zivid_python_library(TYPE SHARED ${ARGN})
endfunction()

function(zivid_python_module_library)
    _zivid_python_library(TYPE MODULE ${ARGN})
endfunction()
