# Zivid Python


Zivid Python is the official Python package for Zivid 3D cameras. Read more about Zivid at [zivid.com](https://www.zivid.com/).

[![Build Status](https://img.shields.io/azure-devops/build/zivid-devops/376f5fda-ba80-4d6c-aaaa-cbcd5e0ad6c0/2/master.svg)](https://dev.azure.com/zivid-devops/zivid-python/_build/latest?definitionId=2&branchName=master) [![PyPI Package](https://img.shields.io/pypi/v/zivid.svg)](https://pypi.org/project/zivid/)


<figure><p align="center"><img src="https://www.zivid.com/hs-fs/hubfs/images/www/ZividOnePlus.jpg?width=500&name=ZividOnePlus.jpg"></p></figure>

---

*Contents:* **[Installation](#installation)** | **[Quick Start](#quick-start)** | **[Examples](#examples)** | **[Versioning](#versioning)** | **[License](#license)** | **[Support](#support)** | **[Test Matrix](#test-matrix)**

---

## Installation

### Dependencies

* [Python](https://www.python.org/) version 3.5 or higher
* [Zivid SDK](https://zivid.atlassian.net/wiki/spaces/ZividKB/pages/59080712/Zivid+Software+Installation) version 1.3.0 or higher
* [Compiler](doc/CompilerInstallation.md) with C++17 support

*Windows users also needs to make sure that the Zivid SDK installation folder is in system `PATH` before using the package, not only the terminal PATH variable. The default install location that should be added to system `PATH` is `C:\Program Files\Zivid\bin`.*

### Using PIP

The easiest way to install the package is to use PIP.

On some systems Python 3 `pip` is called `pip3`. In this guide we assume it is called `pip`.

When using PIP version 19 or higher build dependencies are handled automatically.

Installation may take some time since the `setup.py` script will download additional dependencies and compile C++ source code in the background.

    pip install zivid

#### Old PIP

If you are using a version of PIP older than version 19 please manually install the dependencies listed in [pyproject.toml](pyproject.toml) before installing the zivid.

    pip install scikit-build cmake ninja
    pip install zivid

## Quick Start

Launch a Python interpreter and run the following code.

    import zivid
    app = zivid.Application()
    camera = app.connect_camera()
    frame = camera.capture()
    frame.save("my-frame.zdf")

For more advanced usage see the [Examples](#examples) section.

### Using camera emulation

If you do not have a camera, you can use the `MiscObjects.zdf` file in [ZividSampleData.zip](http://www.zivid.com/software/ZividSampleData.zip) to emulate a camera.

    import zivid
    app = zivid.Application()
    camera = app.create_file_camera("path/to/MiscObjects.zdf")
    frame = camera.capture()
    frame.save("my-frame.zdf")

## Examples

Standalone example programs can be found in the [samples](samples) directory.

## Versioning

This python module is using [PEP 440](https://www.python.org/dev/peps/pep-0440) for versioning. The features available in the module depends on the Zivid SDK version used when building the module. When updating this Python package it is *recommended* to also update to the latest [Zivid SDK](http://www.zivid.com/software). Refer to the [Test Matrix](#test-matrix) for supported version.

The version number of the Zivid Python module consists of six numbers. The three first numbers of the version is the [semantic version](https://semver.org/) of the code in this repository. The last three numbers is the version of the underlying Zivid SDK library used by the Python module.

### Version breakdown

                                        Zivid SDK version = 1.4.1 (semantic version)
                                        v v v
    Zivid Python module version = 1.0.0.1.4.1
                                  ^ ^ ^
                                  Wrapper code version = 1.0.0 (semantic version)

### PyPI

When installing using pip it is possible to specify the required version. This can be useful if upgrading Zivid SDK is not desired, but you want to update Zivid Python.

#### Install latest version of Zivid Python using latest version of Zivid SDK

    pip install zivid

#### Install version 1.0.0 of Zivid Python using latest version of Zivid SDK

    pip install zivid==1.0.0

#### Install version 1.0.0 of Zivid Python using Zivid SDK version 1.4.0

    pip install zivid==1.0.0.1.4.0

#### Install version 1.0.0 of Zivid Python using Zivid SDK version 1.3.0

    pip install zivid==1.0.0.1.3.0
    
*Support for older versions of Zivid SDK will be discontinued when they are no longer compatible with latest version of the wrapper code.*

## License

This project is licensed, see the [LICENSE](LICENSE) file for details.

## Support

Please visit [Zivid Knowledge Base](http://help.zivid.com) for general information on using Zivid 3D cameras. If you cannot find a solution to your issue, please contact support@zivid.com.

## Test matrix

| Operating System | Python version | Zivid SDK version |
| :--------------- | :------------- | :---------------- |
| Ubuntu 18.04     | 3.6            | 1.3.0             |
| Ubuntu 16.04     | 3.5            | 1.3.0             |
| Fedora 30        | 3.7            | 1.3.0             |
| Arch Linux*      | latest         | latest            |
| Windows 10*      | 3.5, 3.6, 3.7  | 1.3.0             |

[*] Only build, no unit testing.
