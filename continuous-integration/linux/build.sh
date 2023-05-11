#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

source $SCRIPT_DIR/venv.sh || exit $?
activate_venv || exit $?

python3 -m pip install --upgrade pip || exit $?
echo "Building zivid-python..."
python3 -m pip install "$ROOT_DIR" || exit $?

echo Success! ["$0"]