#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source ${SCRIPT_DIR}/platform-dependent/common.sh
source ${SCRIPT_DIR}/venv.sh || exit $?

run_platform_dependent_script "setup.sh" || exit $?
create_venv || exit $?

echo Success! [$0]
