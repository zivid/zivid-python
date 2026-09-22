include_guard()

# CMake silently drops the single directory "/usr/include" from the include flags it
# generates, even when an imported target explicitly advertises it. See
# cmLocalGenerator::GetIncludeDirectoriesImplicit() in the CMake sources: when
# CMAKE_<LANG>_IMPLICIT_INCLUDE_DIRECTORIES does not contain "/usr/include" literally, but
# does contain a directory *ending* in "/usr/include", CMake treats the latter as a sysroot
# alias for the former and excludes "/usr/include" for backwards compatibility. The behaviour
# dates back to CMake 3.14 and is unchanged as of 4.4.
#
# The assumption is that the compiler searches "/usr/include" on its own. That does not hold
# for the conda-forge cross-compilers used by conda, mamba and pixi environments
# (x86_64-conda-linux-gnu-g++), whose implicit include list ends in
# "<env>/x86_64-conda-linux-gnu/sysroot/usr/include" and which never look at "/usr/include".
# A Zivid SDK installed system-wide from the .deb packages then becomes invisible:
# find_package(Zivid) succeeds, and every translation unit fails with
# "fatal error: Zivid/<...>.h: No such file or directory".
#
# Re-add the directory with -idirafter, which appends it after the compiler's own directories
# so the sysroot's libc and libstdc++ headers keep priority.
#
# This only ever touches the one path CMake special-cases. An SDK installed under any other
# prefix advertises that prefix instead, CMake emits the flag for it as usual, and this
# function does nothing.
#
# Pass every Zivid target this project compiles against. Any one of them advertising the
# stripped directory is enough to warrant the flag, and the flag is added at most once no
# matter how many do. Targets that do not exist are ignored, so optional components need no
# if(TARGET) guard at the call site.
function(zivid_python_restore_stripped_include_dirs)
    if(MSVC)
        return()
    endif()

    # The only directory CMake hard-excludes.
    set(STRIPPED_DIR "/usr/include")

    if("${STRIPPED_DIR}" IN_LIST CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        # The compiler searches it on its own, so dropping the flag is harmless.
        return()
    endif()

    set(SYSROOT_ALIAS FALSE)
    foreach(IMPLICIT_DIR IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
        if(IMPLICIT_DIR MATCHES "${STRIPPED_DIR}$")
            set(SYSROOT_ALIAS TRUE)
            break()
        endif()
    endforeach()

    if(NOT SYSROOT_ALIAS)
        # CMake emits the flag as usual.
        return()
    endif()

    set(AFFECTED_TARGETS "")
    foreach(TARGET_NAME IN LISTS ARGV)
        if(NOT TARGET ${TARGET_NAME})
            continue()
        endif()
        get_target_property(INCLUDE_DIRS ${TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
        if(INCLUDE_DIRS AND "${STRIPPED_DIR}" IN_LIST INCLUDE_DIRS)
            list(APPEND AFFECTED_TARGETS ${TARGET_NAME})
        endif()
    endforeach()

    if(NOT AFFECTED_TARGETS)
        # The SDK is installed under some other prefix, so nothing is being stripped.
        return()
    endif()

    string(JOIN ", " AFFECTED_TARGETS_TEXT ${AFFECTED_TARGETS})
    message(
        STATUS
        "${AFFECTED_TARGETS_TEXT} advertise ${STRIPPED_DIR}, which CMake strips because the "
        "compiler's implicit include directories contain a sysroot alias for it. Re-adding it "
        "with -idirafter."
    )
    add_compile_options("SHELL:-idirafter ${STRIPPED_DIR}")
endfunction()
