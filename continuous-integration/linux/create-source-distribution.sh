#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

source $SCRIPT_DIR/venv.sh || exit $?
activate_venv || exit $?

# Install minimal requirements to create the source distributions
python3 -m pip install --requirement "$SCRIPT_DIR/../python-requirements/build.txt" || exit $?

# Create source distribution
python3 -m build --sdist "$ROOT_DIR" || exit $?

echo Success! [$0]