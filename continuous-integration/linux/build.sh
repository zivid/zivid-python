#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

pip install --upgrade pip || exit $?
echo "Building zivid-python..."
pip install "$ROOT_DIR" || exit $?

echo Success! ["$0"]