#!/bin/bash

export DEBIAN_FRONTEND=noninteractive
#SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

function apt-yes {
    apt-get --assume-yes "$@"
}

# Setup up Python
apt update || exit $?
apt-yes install python3-pip || exit $?
update-alternatives --install /usr/bin/python python /usr/bin/python3 0 || exit $?
update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 0 || exit $?

source $SCRIPT_DIR/venv.sh || exit $?
create_venv || exit $?
activate_venv || exit $?

# Find and install compatible whl file
pip install --no-index --find-links dist/ zivid || exit $?

echo "SUCCESSFULLY INSTALLED BINARY WHL"
