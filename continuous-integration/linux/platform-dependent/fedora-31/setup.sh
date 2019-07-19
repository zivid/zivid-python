#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Reuse script from previous version as long as that works.
# When needed replace the line below with a new *standalone* script.

$SCRIPT_DIR/../fedora-30/setup.sh $@ || exit $?
