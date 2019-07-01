# Zivid Python


Zivid Python is the official Python package for Zivid 3D cameras. Read more about Zivid at [zivid.com](https://www.zivid.com/).

[![Build Status](https://img.shields.io/azure-devops/build/zivid-devops/376f5fda-ba80-4d6c-aaaa-cbcd5e0ad6c0/2/master.svg)](https://dev.azure.com/zivid-devops/zivid-python/_build/latest?definitionId=2&branchName=master)

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

### Build and install the package

The easiest way to install the package is to use PIP. This package requires [PIP version 19](https://pip.pypa.io/en/stable/installing/#upgrading-pip) or higher.

Installation may take some time since the `setup.py` script will download additional dependencies and compile C++ source code in the background.

#### Linux

    git clone https://github.com/zivid/zivid-python.git
    cd zivid-python
    python -m venv env
    source env/bin/activate
    pip install --upgrade pip
    pip install .

#### Windows

    git clone https://github.com/zivid/zivid-python.git
    cd zivid-python
    python -m venv env
    env\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install .

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

This python module is using [PEP 440](https://www.python.org/dev/peps/pep-0440) for versioning. The features available in the module depends on the Zivid SDK version used  when building the module. When updating this Python package it is *reccomended* to also update to the latest [Zivid SDK](http://www.zivid.com/software). Refer to the [Test Matrix](#test-matrix) for supported version.

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
