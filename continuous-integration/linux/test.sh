#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")
VENV=$(mktemp --tmpdir --directory zivid-python-test-venv-XXXX) || exit $?

python -m venv $VENV || exit $?
source $VENV/bin/activate || exit $?

pip install --upgrade pip || exit $?
pip install "$ROOT_DIR" || exit $?
pip install -r "$SCRIPT_DIR/../requirements-build-and-test.txt" || exit $?

python -m pytest "$ROOT_DIR" -c "$ROOT_DIR/pytest.ini" || exit $?

"$SCRIPT_DIR/run_samples.sh" || exit $?

echo Success! ["$0"]
