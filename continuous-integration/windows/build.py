import sys
import subprocess
from pathlib import Path


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


def _build(root):
    _run_process(("pip", "install", "--verbose", str(root)))


def _main():
    root = Path(__file__).resolve().parents[2]
    _build(root)


if __name__ == "__main__":
    _main()
