#!/bin/bash

# The purpose of this script is to run generate_datamodels.sh in the exact same environment as
# the linting step, so that the generated code will be consistent and pass formatting checks.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

docker run --volume $ROOT_DIR:/host \
           --workdir /host/continuous-integration ubuntu:20.04 \
           bash -c "./linux/setup.sh && ./linux/build.sh && ./code-generation/generate_datamodels.sh" \
           || exit $?

echo Success! ["$0"]