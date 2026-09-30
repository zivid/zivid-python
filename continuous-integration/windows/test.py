import winreg  # pylint: disable=import-error
from os import environ
from pathlib import Path
from common import repo_root, run_process, install_pip_dependencies

# The pytest process dies with a fail-fast exit code that Windows Error Reporting does not
# capture, leaving no stack to work from. Run it under procdump so the next occurrence
# yields one. See ZIVID-14349.
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
    run_process(
        (
            "procdump",
            "-accepteula",
            "-ma",
            "-e",
            "-x",
            str(DUMP_DIR),
            "python",
            "-m",
            "pytest",
            str(root),
            "-c",
            str(root / "pytest.ini"),
        ),
        env=environment,
    )


def _main():
    root = repo_root()
    install_pip_dependencies(
        root / "continuous-integration" / "python-requirements" / "test.txt"
    )
    _test(root)


if __name__ == "__main__":
    _main()
