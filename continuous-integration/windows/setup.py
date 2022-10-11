import os
import tempfile
import shutil
from pathlib import Path
from common import repo_root, run_process, install_pip_dependencies


def _install_zivid_sdk():
    import requests  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as temp_dir:
        zivid_installer_url = "https://www.zivid.com/hubfs/softwarefiles/releases/2.8.0+891708ba-1/windows/ZividSetup_2.8.0+891708ba-1.exe"
        print("Downloading {}".format(zivid_installer_url), flush=True)
        zivid_installer = Path(temp_dir) / "ZividSetup.exe"
        response = requests.get(zivid_installer_url)
        zivid_installer.write_bytes(response.content)
        print("Installing {}".format(zivid_installer), flush=True)
        run_process((str(zivid_installer), "/S"))


def _install_intel_opencl_runtime():
    import requests  # pylint: disable=import-outside-toplevel

    with tempfile.TemporaryDirectory() as temp_dir:
        intel_opencl_runtime_url = "https://www.dropbox.com/s/09bk2nx31hzrupf/opencl_runtime_18.1_x64_setup-20200625-090300.msi?raw=1"
        print("Downloading {}".format(intel_opencl_runtime_url), flush=True)
        opencl_runtime = Path(temp_dir) / "opencl_runtime.msi"
        response = requests.get(intel_opencl_runtime_url)
        opencl_runtime.write_bytes(response.content)
        print("Installing {}".format(opencl_runtime), flush=True)
        run_process(("msiexec", "/i", str(opencl_runtime), "/passive"))


def _write_zivid_cpu_configuration_file():
    api_config = Path() / "ZividAPIConfig.yml"

    appdata_dir = Path(os.getenv("LOCALAPPDATA"))
    target_location = appdata_dir / "Zivid" / "API" / "Config.yml"
    print("Target location for config: " + str(target_location))
    target_location.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(api_config), str(target_location))


def _main():
    root = repo_root()
    install_pip_dependencies(
        root / "continuous-integration" / "python-requirements" / "setup.txt"
    )
    _install_intel_opencl_runtime()
    _install_zivid_sdk()
    _write_zivid_cpu_configuration_file()


if __name__ == "__main__":
    _main()
