import sys
import subprocess
import winreg  # pylint: disable=import-error
from os import environ
from pathlib import Path


def _run_process(args, env=None):
    sys.stdout.flush()
    try:
        process = subprocess.Popen(args, env=env)
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError("Wait failed with exit code {}".format(exit_code))
    except Exception as ex:
        raise type(ex)("Process failed: '{}'.".format(" ".join(args))) from ex
    finally:
        sys.stdout.flush()


def _read_sys_env(environement_variable_name):
    key = winreg.CreateKey(
        winreg.HKEY_LOCAL_MACHINE,
        r"System\CurrentControlSet\Control\Session Manager\Environment",
    )
    return winreg.QueryValueEx(key, environement_variable_name)[0]


def _test(root):
    environment = environ.copy()
    sys_path_key = "PATH"
    sys_path_value = _read_sys_env(sys_path_key)
    environment[sys_path_key] = sys_path_value
    _run_process(
        ("python", "-m", "pytest", str(root), "-c", str(root / "pytest.ini"),),
        env=environment,
    )


def _install_pip_dependencies():
    print("Installing python test requirements", flush=True)
    _run_process(
        (
            "python",
            "-m",
            "pip",
            "install",
            "--requirement",
            "continuous-integration/python-requirements/test.txt",
        )
    )


def _main():
    _install_pip_dependencies()
    root = Path(__file__).resolve().parents[2]
    _test(root)


if __name__ == "__main__":
    _main()
