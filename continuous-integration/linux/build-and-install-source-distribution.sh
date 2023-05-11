#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

source $SCRIPT_DIR/venv.sh || exit $?
activate_venv || exit $?

# Install source distribution
python3 -m pip install --upgrade pip || exit $?
python3 -m pip install "$ROOT_DIR/dist"/*.tar.gz || exit $?

echo Success! [$0]