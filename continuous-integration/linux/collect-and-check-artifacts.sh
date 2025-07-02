#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CI_DIR=$(realpath "$SCRIPT_DIR/..")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

# Check for commit hash argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <commit-hash>"
    exit 1
fi

COMMIT_HASH="$1"

# Simple setup
apt-get update || exit $?
apt-get install --yes python3 python3-venv || exit $?
source $SCRIPT_DIR/venv.sh || exit $?
create_venv || exit $?
activate_venv || exit $?

python3 -m pip install twine || exit $?

# Move artifacts into a common directory
mkdir "$ROOT_DIR/distribution" || exit $?
cp "$ROOT_DIR/artifacts"/*/* "$ROOT_DIR/distribution" || exit $?

# Check against artifact list
python3 "$CI_DIR/deployment/check_expected_artifacts.py" --commit-hash "$COMMIT_HASH" || exit $?

# Check contents of artifacts
twine check "$ROOT_DIR/distribution"/* || exit $?
