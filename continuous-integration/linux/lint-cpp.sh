#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

if [ -z "$1" ]; then
    echo Usage:
    echo $0 build-dir
    exit 1
fi

#Todo: buildDir=$1

cppFiles=$(find "$ROOT_DIR" -name '*.cpp' | grep --invert-match src/3rd-party | grep --invert-match _skbuild | grep --invert-match build)
hFiles=$(find "$ROOT_DIR" -name '*.h' | grep --invert-match src/3rd-party | grep --invert-match _skbuild | grep --invert-match build)

if [ -z "$cppFiles" ] || [ -z "$hFiles" ]; then
    echo Error: Cannot find C++ source files
    exit 1
fi

echo "Checking clang-format conformance"
clang-format --version || exit $?
for fileName in $cppFiles $hFiles; do
    echo $fileName
    diff $fileName \
        <(clang-format $fileName) \
        || exit $?
done

#Todo: echo Building with warnings as errors
#Todo: for compiler in clang gcc; do
#Todo:     source $SCRIPT_DIR/lint-cpp-$compiler-setup.sh || exit $?
#Todo:     echo "Compiler config:"
#Todo:     echo "    CXX:      ${CXX:?} "
#Todo:     echo "    CXXFLAGS: ${CXXFLAGS:?}"
#Todo:     cmake \
#Todo:         -S $ROOT_DIR \
#Todo:         -B $buildDir/$compiler \
#Todo:         -G Ninja \
#Todo:         -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
#Todo:         || exit $?
#Todo:     
#Todo:     cmake --build $buildDir/$compiler || exit $?
#Todo: done
#Todo: 
#Todo: echo Running clang-tidy on C++ files
#Todo: clang-tidy --version || exit $?
#Todo: for fileName in $cppFiles; do
#Todo:     echo $fileName
#Todo:     clang-tidy -p $buildDir/clang $fileName || exit $?
#Todo: done

echo Success! ["$0"]
