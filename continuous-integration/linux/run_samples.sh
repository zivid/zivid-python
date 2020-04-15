#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

SAMPLE_DIR=$(mktemp --tmpdir --directory zivid-python-test-file-camera-XXXX) || exit $?
cd "$SAMPLE_DIR" || exit $?
export PYTHONPATH="$ROOT_DIR" || exit $?

echo "Downloading file camera and sample point cloud"
python "$ROOT_DIR/scripts/sample_data.py" \
    --destination-point-cloud "$SAMPLE_DIR/Zivid3D.zdf" \
    --destination-file-camera "$SAMPLE_DIR/FileCameraZividOne.zfc" \
    || exit $?

for sample in sample_print_version_info \
              sample_capture_from_file
do
    echo Running $sample.py
    python "$ROOT_DIR/samples/$sample.py" || exit $?
done

cd - || exit $?
rm "$SAMPLE_DIR" -r || exit $?

echo Success! ["$0"]
