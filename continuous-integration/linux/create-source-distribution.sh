#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

source $SCRIPT_DIR/venv.sh || exit $?
activate_venv || exit $?

# Install minimal requirements to create the source distributions
python3 -m pip install --requirement "$SCRIPT_DIR/../python-requirements/build.txt" || exit $?

# Create source distribution
# Note: setup.py must be called from the same directory, so we do a local "cd" in a subshell.
(cd $ROOT_DIR && python setup.py sdist) || exit $?

echo Success! [$0]