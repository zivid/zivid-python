import subprocess
import sys
import winreg  # pylint: disable=import-error
from os import environ
from pathlib import Path
from common import repo_root, run_process, install_pip_dependencies

# The pytest process dies with a fail-fast exit code, and neither Windows Error Reporting
# nor procdump's unhandled-exception mode captures that. Only dumping on process
# termination does, so procdump is attached to the pytest process for its whole life and
# the dump is discarded when the run turns out to have been clean. See ZIVID-14349.
DUMP_DIR = Path(r"C:\dumps")


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
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    args = (
        "python",
        "-m",
        "pytest",
        str(root),
        "-c",
        str(root / "pytest.ini"),
    )
    sys.stdout.flush()
    with subprocess.Popen(args, env=environment) as process:
        dumper = subprocess.Popen(
            (
                "procdump",
                "-accepteula",
                "-ma",
                "-t",
                str(process.pid),
                str(DUMP_DIR),
            ),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        exit_code = process.wait()
        dumper.wait()
    sys.stdout.flush()
    if exit_code == 0:
        for dump in DUMP_DIR.glob("*.dmp"):
            dump.unlink()
    else:
        raise RuntimeError("pytest failed with exit code {}".format(exit_code))


def _main():
    root = repo_root()
    install_pip_dependencies(
        root / "continuous-integration" / "python-requirements" / "test.txt"
    )
    _test(root)


if __name__ == "__main__":
    _main()
