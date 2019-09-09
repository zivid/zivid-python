import argparse
import tempfile
import sys
import subprocess
from pathlib import Path
import requests


def _options():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", required=True, help="The repository root", type=Path)
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip setting up build and test dependencies",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip the build step")
    parser.add_argument(
        "--skip-test", action="store_true", help="Skip the unit test step"
    )

    return parser.parse_args()


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


def _setup():
    with tempfile.TemporaryDirectory() as temp_dir:
        zivid_installer_url = "https://www.zivid.com/hubfs/softwarefiles/releases/1.5.0+63f281e2-26/windows/ZividSetup_1.5.0+63f281e2-26.exe"
        print("Downloading {}".format(zivid_installer_url), flush=True)
        zivid_installer = Path(temp_dir) / "ZividSetup.exe"
        response = requests.get(zivid_installer_url)
        zivid_installer.write_bytes(response.content)
        print("Installing {}".format(zivid_installer), flush=True)
        _run_process((str(zivid_installer), "/S"))


def _build(root):
    _run_process(("pip", "install", str(root)))


def _test(root):
    _run_process(
        (
            "echo",
            "TODO: ",
            "python",
            "-m",
            "pytest",
            str(root),
            "-c",
            str(root / "pytest.ini"),
        )
    )


def _main():
    options = _options()

    if not options.skip_setup:
        _setup()

    if not options.skip_build:
        _build(options.root)

    if not options.skip_test:
        _test(options.root)


if __name__ == "__main__":
    _main()
