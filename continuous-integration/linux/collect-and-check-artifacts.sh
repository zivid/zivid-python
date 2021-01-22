#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CI_DIR=$(realpath "$SCRIPT_DIR/..")
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

# Simple setup
apt-get update || exit $?
apt-get install --yes python3 python3-pip || exit $?
pip3 install twine || exit $?

# Move artifacts into a common directory
mkdir "$ROOT_DIR/distribution" || exit $?
cp "$ROOT_DIR/artifacts"/*/* "$ROOT_DIR/distribution" || exit $?

# Check against artifact list
python3 "$CI_DIR/deployment/check_expected_artifacts.py" || exit $?

# Check contents of artifacts
twine check "$ROOT_DIR/distribution"/* || exit $?
