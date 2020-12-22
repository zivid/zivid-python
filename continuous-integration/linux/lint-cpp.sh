#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

if [ -z "$1" ]; then
    echo Usage:
    echo $0 build-dir
    exit 1
fi


cppFiles=$(find "$ROOT_DIR" -name '*.cpp' |grep -v src/3rd-party)
hFiles=$(find "$ROOT_DIR" -name '*.h' |grep -v src/3rd-party)

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

buildDir=$1
SDK_VERSION = $(echo -e "from setup import zivid_sdk_version\nprint(zivid_sdk_version())" | python)

echo Building with warnings as errors
for compiler in clang, gcc; do
    source $SCRIPT_DIR/lint-cpp-$compiler-setup.sh || exit $?
    echo "Compiler config:"
    echo "    CXX:      ${CXX:?} "
    echo "    CXXFLAGS: ${CXXFLAGS:?}"
    cmake \
        -S $ROOT_DIR \
        -B $buildDir/$compiler \
        -G Ninja \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DCMAKE_BUILD_TYPE=Debug \
        -DZIVID_SDK_VERSION=$SDK_VERSION \
        || exit $?
    
    cmake --build $buildDir/$compiler || exit $?
done
#Todo: 
#Todo: echo Running clang-tidy on C++ files
#Todo: clang-tidy --version || exit $?
#Todo: for fileName in $cppFiles; do
#Todo:     echo $fileName
#Todo:     clang-tidy -p $buildDir/clang $fileName || exit $?
#Todo: done

echo Success! ["$0"]
