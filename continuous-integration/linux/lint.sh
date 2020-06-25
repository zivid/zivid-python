#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

pythonFiles=$(find "$ROOT_DIR" -name '*.py' -not -path '*doc/scratchpad*')
publicPythonFiles=$(find "$ROOT_DIR/modules" "$ROOT_DIR/samples" -name '*.py')
nonPublicPythonFiles=$(comm -23 <(echo $pythonFiles| tr " " "\n" |sort) \
                                <(echo $publicPythonFiles| tr " " "\n" |sort))

bashFiles=$(find "$ROOT_DIR" -name '*.sh')

pip install --requirement "$SCRIPT_DIR/../python-requirements/lint.txt" || exit $?

echo Running non public pylint on:
echo "$nonPublicPythonFiles"
pylint \
    --rcfile "$ROOT_DIR/.pylintrc" \
    --extension-pkg-whitelist=_zivid \
    --generated-members=_zivid.* \
    $nonPublicPythonFiles \
    || exit $?

echo Running public pylint on:
echo "$publicPythonFiles"
pylint \
    --rcfile "$ROOT_DIR/.pylintrc-packaged-files" \
    $publicPythonFiles \
    || exit $?

echo Running non public flake8 on:
echo "$nonPublicPythonFiles"
flake8 --config="$ROOT_DIR/.flake8" $nonPublicPythonFiles || exit $?

echo Running public flake8 on:
echo "$publicPythonFiles"
flake8 --config="$ROOT_DIR/.flake8-packaged-files" $publicPythonFiles || exit $?

echo Running darglint on:
echo "$publicPythonFiles"
darglint $publicPythonFiles || exit $?

echo Running black on:
echo "$pythonFiles"
black --check --diff $pythonFiles || exit $?

echo Running shellcheck on:
echo "$bashFiles"
shellcheck -x -e SC1090,SC2086,SC2046 $bashFiles || exit $?

echo Running code analysis on C++ code:
CPP_LINT_DIR=$(mktemp --tmpdir --directory zivid-python-cpp-lint-XXXX) || exit $?
$SCRIPT_DIR/lint-cpp.sh $CPP_LINT_DIR || exit $?
rm -r $CPP_LINT_DIR || exit $?

echo Success! ["$0"]

