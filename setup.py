import tempfile
import platform
import subprocess
from sys import version_info
from pathlib import Path
from pkgutil import iter_modules
from skbuild import setup

# To be replaced by: from setuptools_scm import get_version
def get_version():
    return "2.1.2"


def _zivid_sdk_version():
    return "2.4.1"


def _zivid_python_version():
    scm_version = get_version()

    if "+" in scm_version:
        base_version, scm_metadata = scm_version.split("+", 1)
    else:
        base_version = scm_version
        scm_metadata = None

    base_version = "{}.{}".format(base_version, _zivid_sdk_version())

    if scm_metadata:
        version = "{}+{}".format(base_version, scm_metadata)
    else:
        version = base_version

    return version


def _python_version():
    return "{}.{}.{}".format(*version_info)


def _make_message_box(*message):
    width = max([len(e) for e in message])

    box_bar = "+-" + "-" * width + "-+"
    empty_line = "\n| " + " " * width + " |\n"
    message_lines = ["| " + line + " " * (width - len(line)) + " |" for line in message]

    return (
        "\n\n" + box_bar + "\n" + empty_line.join(message_lines) + "\n" + box_bar + "\n"
    )


def _check_dependency(module_name, package_hint=None):
    if package_hint is None:
        package_hint = module_name
    if module_name not in [module[1] for module in iter_modules()]:
        raise ImportError(
            _make_message_box(
                "!! Missing module '{}' !!".format(module_name),
                "Please install '{}' manually or use PIP>=19 to handle build dependencies automatically (PEP 517)".format(
                    package_hint
                ),
            )
        )


def _check_cpp17_compiler():
    def run_process(args, **kwargs):
        try:
            process = subprocess.Popen(args, **kwargs)
            exit_code = process.wait()
            if exit_code != 0:
                raise RuntimeError("Wait failed with exit code {}".format(exit_code))
        except Exception as ex:
            raise type(ex)("Process failed: '{}'.".format(" ".join(args))) from ex

    try:
        run_process(("cmake", "--version"))
    except Exception as ex:
        raise RuntimeError(_make_message_box("!! CMake not found !!")) from ex
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
    _check_dependency("conans", "conan")
    _check_dependency("ninja")
    _check_dependency("skbuild", "scikit-build")

    _check_cpp17_compiler()

    setup(
        name="zivid",
        version=_zivid_python_version(),
        description="Defining the Future of 3D Machine Vision",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://www.zivid.com",
        author="Zivid AS",
        author_email="customersuccess@zivid.com",
        license="BSD 3-Clause",
        packages=["zivid", "zivid._calibration", "_zivid"],
        package_dir={"": "modules"},
        install_requires=["numpy"],
        cmake_args=[
            "-DZIVID_PYTHON_VERSION=" + _zivid_python_version(),
            "-DZIVID_SDK_VERSION=" + _zivid_sdk_version(),
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
