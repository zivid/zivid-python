from skbuild import setup


# To be replaced by: from setuptools_scm import get_version
def get_version():
    return "0.9.0"


def _zivid_sdk_version():
    return "1.3.0"


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


setup(
    name="zivid",
    version=_zivid_python_version(),
    description="Defining the Future of 3D Machine Vision",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://www.zivid.com",
    author="Zivid AS",
    author_email="support@zivid.com",
    license=open("LICENSE").read(),
    packages=["zivid", "_zivid"],
    package_dir={"": "modules"},
    install_requires=["numpy"],
    cmake_args=[
        "-DZIVID_PYTHON_VERSION=" + _zivid_python_version(),
        "-DZIVID_SDK_VERSION=" + _zivid_sdk_version(),
        "-Dpybind11_DIR=src/3rd-party/pybind11-2.2.4/share/cmake/pybind11/",
    ],
)
