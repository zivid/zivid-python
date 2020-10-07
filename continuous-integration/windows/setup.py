import tempfile
import sys
import subprocess
import shutil
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
        zivid_installer_url = "https://www.zivid.com/hubfs/softwarefiles/releases/2.1.0+d2007e12-1/windows/ZividSetup_2.1.0+d2007e12-1.exe"
        print("Downloading {}".format(zivid_installer_url), flush=True)
        zivid_installer = Path(temp_dir) / "ZividSetup.exe"
        response = requests.get(zivid_installer_url)
        zivid_installer.write_bytes(response.content)
        print("Installing {}".format(zivid_installer), flush=True)
        _run_process((str(zivid_installer), "/S"))


def _install_intel_opencl_runtime():
    import requests  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as temp_dir:
        intel_opencl_runtime_url = "https://www.dropbox.com/s/09bk2nx31hzrupf/opencl_runtime_18.1_x64_setup-20200625-090300.msi?raw=1"
        print("Downloading {}".format(intel_opencl_runtime_url), flush=True)
        opencl_runtime = Path(temp_dir) / "opencl_runtime.msi"
        response = requests.get(intel_opencl_runtime_url)
        opencl_runtime.write_bytes(response.content)
        print("Installing {}".format(opencl_runtime), flush=True)
        _run_process(("msiexec", "/i", str(opencl_runtime), "/passive"))


def _write_zivid_cpu_configuration_file():
    api_config = Path() / "ZividAPIConfig.yml"
    target_location = Path(
        r"C:\Users\VssAdministrator\AppData\Local\Zivid\API\Config.yml"
    )
    target_location.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(api_config), str(target_location))


def _main():
    _install_pip_dependencies()
    _install_intel_opencl_runtime()
    _install_zivid_sdk()
    _write_zivid_cpu_configuration_file()


if __name__ == "__main__":
    _main()
