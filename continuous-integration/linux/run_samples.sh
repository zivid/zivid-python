#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

SAMPLE_DIR=$(mktemp --tmpdir --directory zivid-python-test-file-camera-XXXX) || exit $?
cd "$SAMPLE_DIR" || exit $?
export PYTHONPATH="$ROOT_DIR" || exit $?

echo "Downloading MiscObjects.zdf"
python "$ROOT_DIR/scripts/sample_data.py" --destination "$SAMPLE_DIR/MiscObjects.zdf" || exit $?

echo "Downloading samples from python-samples repository"
python "$ROOT_DIR/scripts/get_samples.py" --destination "$SAMPLE_DIR" || exit $?

for sample in print_version_info \
              capture_from_file
do
    echo Running $sample.py
    python "$sample.py" || exit $?
done

cd - || exit $?
rm "$SAMPLE_DIR" -r || exit $?

echo Success! ["$0"]
