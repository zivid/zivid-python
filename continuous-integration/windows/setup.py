import tempfile
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


def _install_pip_dependencies():
    print("Installing python build requirements", flush=True)
    _run_process(
        (
            "pip",
            "install",
            "--requirement",
            "continuous-integration/python-requirements/setup.txt",
        )
    )


def _install_zivid_sdk():
    import requests  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as temp_dir:
        zivid_installer_url = "https://www.zivid.com/hubfs/softwarefiles/releases/1.8.1+6967bc1b-1/windows/ZividSetup_1.8.1+6967bc1b-1.exe"
        print("Downloading {}".format(zivid_installer_url), flush=True)
        zivid_installer = Path(temp_dir) / "ZividSetup.exe"
        response = requests.get(zivid_installer_url)
        zivid_installer.write_bytes(response.content)
        print("Installing {}".format(zivid_installer), flush=True)
        _run_process((str(zivid_installer), "/S"))


def _main():
    _install_pip_dependencies()
    _install_zivid_sdk()


if __name__ == "__main__":
    _main()
