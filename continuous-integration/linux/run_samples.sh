#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

SAMPLE_DIR=$(mktemp --tmpdir --directory zivid-python-test-file-camera-XXXX) || exit $?
cd "$SAMPLE_DIR" || exit $?
export PYTHONPATH="$ROOT_DIR" || exit $?

echo "Downloading MiscObjects.zdf"
python "$ROOT_DIR/scripts/sample_data.py" --destination "$SAMPLE_DIR/MiscObjects.zdf" || exit $?

for sample in sample_print_version_info \
              sample_capture_from_file
do
    echo Running $sample.py
    python "$ROOT_DIR/samples/$sample.py" || exit $?
done

cd - || exit $?
rm "$SAMPLE_DIR" -r || exit $?

echo Success! ["$0"]
