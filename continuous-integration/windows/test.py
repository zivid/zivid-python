import argparse
import tempfile
import sys
import subprocess
from pathlib import Path
import requests


def _options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="The repository root", type=Path)
    return parser.parse_args()


def _run_process(args):
    sys.stdout.flush()
    try:
        process = subprocess.Popen(args)
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError("Wait failed with exit code {}".format(exit_code))
    except Exception as ex:
        raise type(ex)("Process failed: '{}'.".format(" ".join(args))) from ex
    finally:
        sys.stdout.flush()


def _test(root):
    _run_process(
        (
            "echo",
            "TODO: ",
            "python",
            "-m",
            "pytest",
            str(root),
            "-c",
            str(root / "pytest.ini"),
        )
    )


def _main():
    options = _options()

    _test(options.root)


if __name__ == "__main__":
    _main()
