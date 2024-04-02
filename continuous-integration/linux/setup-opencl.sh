#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

source ${SCRIPT_DIR}/platform-dependent/common.sh
run_platform_dependent_script "setup-opencl.sh" || exit $?

echo "clinfo:"
clinfo || exit $?

install -D "$ROOT_DIR"/ZividAPIConfig.yml "$HOME"/.config/Zivid/API/Config.yml || exit $?
