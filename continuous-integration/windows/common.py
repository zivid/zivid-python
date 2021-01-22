import sys
import subprocess
from pathlib import Path


def repo_root():
    return Path(__file__).resolve().parents[2]


def run_process(args, env=None, workdir=None):
    sys.stdout.flush()
    try:
        process = subprocess.Popen(args, env=env, cwd=workdir)
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError("Wait failed with exit code {}".format(exit_code))
    except Exception as ex:
        raise type(ex)("Process failed: '{}'.".format(" ".join(args))) from ex
    finally:
        sys.stdout.flush()


def install_pip_dependencies(requirements_file):
    print("Installing python test requirements", flush=True)
    run_process(
        ("python", "-m", "pip", "install", "--requirement", str(requirements_file),)
    )
