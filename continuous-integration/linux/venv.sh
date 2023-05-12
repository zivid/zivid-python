#!/bin/bash

export VENV_PATH=/tmp/zivid-python-ci-venv

function create_venv {
    echo "Creating venv"
    python3 -m venv ${VENV_PATH} || exit $?
}

function activate_venv {
    echo "Activating venv"
    # shellcheck disable=SC1091
    source ${VENV_PATH}/bin/activate || exit $?
}
