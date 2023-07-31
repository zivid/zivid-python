#!/bin/bash

# For consistent results, there are two options:
# - Run this script in the same Docker container that runs linting (docker_generate_datamodels.sh)
# or
# - Start a fresh virtual environment, install zivid-python from source, and run this script.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

if [ -z "$VIRTUAL_ENV" ]; then
    source $SCRIPT_DIR/../linux/venv.sh || exit $?
    activate_venv || exit $?
fi

python3 -m pip install --requirement "$SCRIPT_DIR/../python-requirements/lint.txt" || exit $?

python3 "$ROOT_DIR/continuous-integration/code-generation/datamodel_frontend_generator.py" \
    --dest-dir "$ROOT_DIR/modules/zivid/" \
    || exit $?

echo Success! ["$0"]