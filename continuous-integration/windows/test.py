import winreg  # pylint: disable=import-error
from os import environ
from common import repo_root, run_process, install_pip_dependencies


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
    run_process(
        (
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
