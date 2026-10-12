#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${SCRIPT_DIR}/platform-dependent/common.sh
source ${SCRIPT_DIR}/versions.sh
run_platform_dependent_script "install-sdk.sh" || exit $?
