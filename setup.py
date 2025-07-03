import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from pkgutil import iter_modules
from sys import version_info
from tempfile import TemporaryDirectory

from packaging.utils import canonicalize_version
from pkginfo.sdist import UnpackedSDist
from skbuild import constants, setup


def _read_sdk_version_json():
    sdk_version_path = Path(__file__).parent / "sdk_version.json"
    with sdk_version_path.open("r") as f:
        return json.load(f)


def _determine_package_version():
    sdk_version = _read_sdk_version_json()

    version_segments = [
        str(sdk_version["major"]),
        str(sdk_version["minor"]),
        str(sdk_version["patch"]),
    ]

    github_repository = os.getenv("GITHUB_REPOSITORY")
    github_ref = os.getenv("GITHUB_REF")

    local_version_segments = []

    if sdk_version.get("pre_release"):
        local_version_segments.append(sdk_version["pre_release"])

    commit_hash = os.getenv("CI_COMMIT_HASH")
    github_sha = os.getenv("GITHUB_SHA")

    commit_hash = commit_hash or github_sha
    if commit_hash is not None:
        hash_length = 8
        local_version_segments.append(commit_hash[:hash_length])
    else:
        local_version_segments.append("unofficial")

    if github_repository != "zivid/zivid-python" or github_ref != "refs/heads/master":
        # Only the master branch of the zivid-python repository is considered stable.
        # Anywhere else the version will be a development version, and we add the local version.
        version_segments.append("dev0")
        version = ".".join(version_segments) + "+" + ".".join(local_version_segments)
    else:
        # Stable releases must not include a local version to be allowed to upload to PyPI.
        version = ".".join(version_segments)


    return canonicalize_version(version, strip_trailing_zero=False)


def _get_version():
    try:
        d = UnpackedSDist(__file__)
        return d.version
    except ValueError:
        return _determine_package_version()


def _python_version():
    return "{}.{}.{}".format(*version_info)


def _make_message_box(*message):
    width = max(len(e) for e in message)

    box_bar = "+-" + "-" * width + "-+"
    empty_line = "\n| " + " " * width + " |\n"
    message_lines = ["| " + line + " " * (width - len(line)) + " |" for line in message]

    return "\n\n" + box_bar + "\n" + empty_line.join(message_lines) + "\n" + box_bar + "\n"


def _check_dependency(module_name, package_hint=None):
    if package_hint is None:
        package_hint = module_name
    if module_name not in [module[1] for module in iter_modules()]:
        raise ImportError(
            _make_message_box(
                "!! Missing module '{}' !!".format(module_name),
                "Please install '{}' manually or use PIP>=19 to handle build dependencies automatically"
                " (PEP 517)".format(package_hint),
            )
        )


def _check_cpp17_compiler():
    def run_process(args, **kwargs):
        try:
            with subprocess.Popen(args, **kwargs) as process:
                exit_code = process.wait()
                if exit_code != 0:
                    raise RuntimeError("Wait failed with exit code {}".format(exit_code))
        except Exception as ex:
            raise type(ex)("Process failed: '{}'.".format(" ".join(args))) from ex

    try:
        run_process(("cmake", "--version"))
    except Exception as ex:
        raise RuntimeError(
            _make_message_box(
                "!! CMake not found !!",
                "Ensure PIP is updated to the latest version to handle build dependencies automatically,",
                "or install CMake and Ninja manually.",
            )
        ) from ex
    with tempfile.TemporaryDirectory(prefix="zivid-python-build-") as temp_dir:
        with (Path(temp_dir) / "lib.cpp").open("w") as lib_cpp:
            # MSVC does not report itself as C++17, on Windoes we have to rely on the CMAKE_CXX_STANDARD test below
            if platform.system() == "Linux":
                lib_cpp.write("static_assert(__cplusplus >= 201703L);")
        with (Path(temp_dir) / "CMakeLists.txt").open("w") as cmake_lists_txt:
            cmake_lists_txt.write(
                "project(zivid-python-compiler-detection LANGUAGES CXX)\n"
                "set(CMAKE_CXX_STANDARD 17)\n"
                "add_library(lib lib.cpp)\n"
            )
        try:
            if platform.system() == "Linux":
                run_process(("cmake", "-GNinja", "."), cwd=temp_dir)
            else:
                run_process(("cmake", "."), cwd=temp_dir)
            run_process(("cmake", "--build", "."), cwd=temp_dir)
        except Exception as ex:
            raise RuntimeError(
                _make_message_box(
                    "!! Module setup failed !!",
                    "Make sure you have a working C++17 compiler installed",
                    "Refer to Readme.md for detailed installation instructions",
                )
            ) from ex


def _main():
    # This list is a duplicate of the build-system requirements in pyproject.toml.
    # The purpose of these checks is to help users with PIP<19 lacking support for
    # pyproject.toml
    # Keep the two lists in sync
    _check_dependency("cmake")
    _check_dependency("conan")
    _check_dependency("ninja")
    _check_dependency("skbuild", "scikit-build")
    _check_dependency("pkginfo")

    _check_cpp17_compiler()

    with TemporaryDirectory(prefix="zivid-python-build_") as build_dir:
        print("Overriding build dir: " + build_dir)
        constants.SKBUILD_DIR = lambda: build_dir

        version = _get_version()

        setup(
            name="zivid",
            version=version,
            description="Defining the Future of 3D Machine Vision",
            long_description=Path("README.md").read_text(encoding="utf-8"),
            long_description_content_type="text/markdown",
            url="https://www.zivid.com",
            author="Zivid AS",
            author_email="customersuccess@zivid.com",
            license="BSD 3-Clause",
            packages=[
                "zivid",
                "zivid._calibration",
                "zivid.experimental",
                "zivid.experimental.point_cloud_export",
                "zivid.experimental.toolbox",
                "_zivid",
            ],
            package_dir={"": "modules"},
            install_requires=["numpy"],
            cmake_args=[
                "-DZIVID_PYTHON_VERSION=" + version,
                "-DPYTHON_INTERPRETER_VERSION=" + _python_version(),
            ],
            classifiers=[
                "License :: OSI Approved :: BSD License",
                "Programming Language :: Python :: 3",
                "Programming Language :: Python :: Implementation :: CPython",
                "Topic :: Scientific/Engineering",
            ],
        )


if __name__ == "__main__":
    _main()
