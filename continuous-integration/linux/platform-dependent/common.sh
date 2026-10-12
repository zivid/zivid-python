#!/bin/bash

PLATFORM_DEPENDENT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

function run_platform_dependent_script {
    source /etc/os-release || exit $?
    ${PLATFORM_DEPENDENT_DIR}/${ID}/${1}
}
