#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

function apt-yes {
    apt-get --assume-yes "$@"
}

# Setup up Python
export DEBIAN_FRONTEND=noninteractive
apt update || exit $?
apt-yes install python3-pip python3-venv || exit $?
update-alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?

source $SCRIPT_DIR/venv.sh || exit $?
create_venv || exit $?
activate_venv || exit $?

# Find and install compatible whl file
tree ${ROOT_DIR}/dist/
python3 -m pip install --upgrade pip || exit $?
python3 -m pip install numpy || exit $?
python3 -m pip install --no-index --find-links ${ROOT_DIR}/dist/ zivid || exit $?
echo "SUCCESSFULLY INSTALLED BINARY WHL"
