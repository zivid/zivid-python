#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR=$(realpath "$SCRIPT_DIR/../..")

python3 -m pip install \
    --requirement "$SCRIPT_DIR/../python-requirements/lint.txt" ||
    exit $?


runPylint() {
    local fileList="$1"
    local rcfile=$ROOT_DIR/"$2"
    echo ""
    echo "Running pylint with ${rcfile} on:"
    echo "${fileList}"
    pylint --rcfile "$rcfile" $fileList || exit $?
}

runFlake8() {
    local fileList="$1"
    local rcfile=$ROOT_DIR/"$2"
    echo ""
    echo "Running flake8 with ${rcfile} on:"
    echo "${fileList}"
    flake8 --config="$rcfile" $fileList || exit $?
}

runDarglint() {
    local fileList="$1"
    echo ""
    echo "Running darglint on:"
    echo "${fileList}"
    darglint $fileList || exit $?
}

runBlackCheck() {
    local fileList="$1"
    echo ""
    echo "Running black on:"
    echo "${fileList}"
    black --check --diff $fileList || exit $?
}

runShellcheck() {
    local fileList="$1"
    echo ""
    echo "Running shellcheck on:"
    echo "${fileList}"
    shellcheck -x -e SC1090,SC2086,SC2046 $fileList || exit $?
}

# Get list of all bash files
bashFiles=$(find "$ROOT_DIR" -name '*.sh')

# Get list of all Python files considered public (in certain directories, without leading underscore)
publicPythonFiles=$(find "$ROOT_DIR/modules" "$ROOT_DIR/samples" -regex '.*\/[^_]\w+\.py$')
# Divide all Python files into public and non-public
allPythonFiles=$(find "$ROOT_DIR" -name '*.py' -not -path '*doc/scratchpad*')
nonPublicPythonFiles=$(comm -23 <(echo $allPythonFiles| tr " " "\n" |sort) \
                                <(echo $publicPythonFiles| tr " " "\n" |sort))
# Separate out test-related files. Any remaining Python files are then counted as "internal"
testsPythonFiles=$(find "$ROOT_DIR/test" -name '*.py')
internalPythonFiles=$(comm -23 <(echo $nonPublicPythonFiles| tr " " "\n" |sort) \
                               <(echo $testsPythonFiles| tr " " "\n" |sort))

# Check that generated datamodel front-ends are up to date
python3 "$ROOT_DIR/continuous-integration/code-generation/check_datamodels_up_to_date.py" || exit $?

# Python linting
runPylint "$publicPythonFiles" ".pylintrc" || exit $?
runPylint "$internalPythonFiles" ".pylintrc-internal" || exit $?
runPylint "$testsPythonFiles" ".pylintrc-tests" || exit $?

runFlake8 "$publicPythonFiles" ".flake8" || exit $?
runFlake8 "$internalPythonFiles" ".flake8-internal" || exit $?
runFlake8 "$testsPythonFiles" ".flake8-tests" || exit $?

runDarglint "$publicPythonFiles" || exit $?

runBlackCheck "$allPythonFiles" || exit $?

# Shell script linting
runShellcheck "$bashFiles"

# C++ linting
echo "Running code analysis on C++ code:"
CPP_LINT_DIR=$(mktemp --tmpdir --directory zivid-python-cpp-lint-XXXX) || exit $?
$SCRIPT_DIR/lint-cpp.sh $CPP_LINT_DIR || exit $?
rm -r $CPP_LINT_DIR || exit $?

echo Success! ["$0"]

