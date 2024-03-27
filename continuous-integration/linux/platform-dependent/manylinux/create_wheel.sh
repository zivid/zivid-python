#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../../../..")

if [ $# -eq 0 ]; then
    echo "Usage: $0 <target (e.g. cp311)>"
    exit 1
fi
TARGET=$1
MANYLINUX_PLATFORM="manylinux_2_35_x86_64"
FULL_PYTHON_TARGET="${TARGET}-${TARGET}"
TMPDIR="/tmp/zivid-python-wheelhouse/"

echo "Making ${MANYLINUX_PLATFORM} for ${TARGET}"
PYBIN="/opt/python/${FULL_PYTHON_TARGET}/bin"
${PYBIN}/pip wheel ${ROOT_DIR} --no-deps -w ${TMPDIR} || exit $?

for wheel in "${TMPDIR}"/zivid-*-"${FULL_PYTHON_TARGET}"-*.whl; do
    auditwheel repair ${wheel} --plat ${MANYLINUX_PLATFORM} -w ${ROOT_DIR}/dist/ || exit $?
done
