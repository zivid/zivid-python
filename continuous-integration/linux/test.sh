#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

python3_minor_version=$(python3 -c 'import platform; print(platform.python_version_tuple()[1])') || exit $?

if [ $python3_minor_version == "6" ]; then
    # Use legacy requirements file because latest packages do not support Python 3.6.
    # Remove this once we drop support for Python 3.6 on Linux. 
    test_requirements_file="python-requirements/legacy-python36/test.txt"
    echo "Detected Python 3.6. Using legacy requirements file: ${test_requirements_file}"
else
    test_requirements_file="python-requirements/test.txt"
fi

python3 -m pip install --requirement "$SCRIPT_DIR/../${test_requirements_file}" || exit $?

python -m pytest "$ROOT_DIR" -c "$ROOT_DIR/pytest.ini" || exit $?

echo Success! ["$0"]
