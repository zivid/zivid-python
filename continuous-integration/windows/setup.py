import json
import os
import tempfile
import shutil
from pathlib import Path
from common import repo_root, run_process, install_pip_dependencies


# On Windows the TemporaryDirectory cleanup can sometimes fail.
# In Python 3.10 and newer, this class has a new argument called
# `ignore_cleanup_errors` that would help us workaround this issue.
# However, until we can use Python 3.10, we will need to implement
# our own workaround.
class TemporaryDirectoryIgnoringCleanupErrors:
    def __init__(self):
        self._tmp_dir = (
            tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        )

    def __enter__(self):
        return self._tmp_dir.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._tmp_dir.cleanup()
        except OSError:
            pass
        finally:
            self._tmp_dir = None

    def __fspath__(self):
        return self.name

    @property
    def name(self):
        return self._tmp_dir.name if self._tmp_dir else None


def _install_zivid_sdk():
    import requests  # pylint: disable=import-outside-toplevel

    versions_json = (Path(__file__).parents[1] / "versions.json").read_text(
        encoding="utf8"
    )
    exact_version = json.loads(versions_json)["ZIVID_SDK_EXACT_VERSION"]

    with TemporaryDirectoryIgnoringCleanupErrors() as temp_dir:
        zivid_installer_url = f"https://downloads.zivid.com/sdk/releases/{exact_version}/windows/ZividSetup_{exact_version}.exe"
        print("Downloading {}".format(zivid_installer_url), flush=True)
        zivid_installer = Path(temp_dir) / "ZividSetup.exe"
        response = requests.get(zivid_installer_url, timeout=(25, 600))
        zivid_installer.write_bytes(response.content)
        print("Installing {}".format(zivid_installer), flush=True)
        run_process(
            (
                str(zivid_installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/TYPE=full",
            )
        )


def _install_intel_opencl_runtime():
    import requests  # pylint: disable=import-outside-toplevel

    with TemporaryDirectoryIgnoringCleanupErrors() as temp_dir:
        intel_oneapi_opencl_installer_url = "https://registrationcenter-download.intel.com/akdlm/IRC_NAS/faf10bb4-a1b3-46cf-ae0b-986b419e1b1c-opencl/w_opencl_runtime_p_2023.2.0.49500.exe"
        print("Downloading {}".format(intel_oneapi_opencl_installer_url), flush=True)
        opencl_runtime_installer = Path(temp_dir) / "opencl_runtime_installer.exe"
        response = requests.get(intel_oneapi_opencl_installer_url, timeout=(25, 600))
        opencl_runtime_installer.write_bytes(response.content)
        print("Installing {}".format(opencl_runtime_installer), flush=True)
        run_process((str(opencl_runtime_installer), "--silent", "--a", "/quiet"))


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
