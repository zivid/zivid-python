#!/bin/bash

SCRIPT_DIR="$(realpath $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) )"

# Read versions.json and set as environment variables
VERSIONS_FILE="${SCRIPT_DIR}/../versions.json"
for var in $(jq -r "to_entries|map(\"\(.key)=\(.value|tostring)\")|.[]" ${VERSIONS_FILE} ); do
    echo "Setting env var from ${VERSIONS_FILE}: ${var}"
    export ${var?}
done
