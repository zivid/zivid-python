#!/bin/bash

# This script uses info in /etc/os-release to dispatch setup to a
# platform specific implementation

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

# Elevate permissions
if [ $EUID != 0 ]; then
    sudo "$0" "$@"
    exit $?
fi

source /etc/os-release || exit $?

osId=$ID-$VERSION_ID

setupScript=$SCRIPT_DIR/platform-dependent/$osId/setup.sh

if [[ -f $setupScript ]]; then
    $setupScript || exit $?
else
    echo $setupScript not found
    echo Support for $PRETTY_NAME is not implemented
    exit 1
fi

echo "clinfo:"
clinfo || exit $?

install -D "$ROOT_DIR"/ZividAPIConfig.yml "$HOME"/.config/Zivid/API/Config.yml || exit $?

source $SCRIPT_DIR/venv.sh || exit $?
create_venv || exit $?

echo Success! [$0]
