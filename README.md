# Zivid Python

Zivid Python is the official Python package for Zivid 3D cameras. Read more about Zivid at [zivid.com](https://www.zivid.com/).

[![Build Status][ci-badge]][ci-url] [![PyPI Package][pypi-badge]][pypi-url]
![Zivid Image][header-image]

---

*Contents:* **[Installation](#installation)** | **[Quick Start](#quick-start)** | **[Examples](#examples)** | **[Versioning](#versioning)** | **[License](#license)** | **[Support](#support)** | **[Test Matrix](#test-matrix)**

---

## Installation

### Dependencies

* [Python](https://www.python.org/) version 3.7 or higher
* [Zivid SDK][zivid-download-software-url] version 2.14.0 (see [here][zivid-software-installation-url] for help)
* [Compiler](doc/CompilerInstallation.md) with C++17 support

*Ubuntu users must install Python headers (`apt install python3-dev`) in addition to the regular `python3` package.*

*Windows users also needs to make sure that the Zivid SDK installation folder is in system `PATH` before using the package, not only the terminal PATH variable. The default install location that should be added to system `PATH` is `C:\Program Files\Zivid\bin`.*

### Installing official version from PyPI using PIP

After having installed the latest Zivid SDK, the easiest way to install Zivid Python is to use PIP to fetch the latest official version from PyPI:

    pip install zivid

Note:

> If you don't use the latest Zivid SDK version you need to manually specify the version. See [Versioning](#versioning).

Installation may take some time since the `setup.py` script will download additional dependencies and compile C++ source code in the background.

On some systems Python 3 `pip` is called `pip3`. In this guide we assume it is called `pip`. When using PIP version 19 or higher build dependencies are handled automatically.

#### Old PIP

If you are using a version of PIP older than version 19 please manually install the dependencies listed in [pyproject.toml](pyproject.toml) before installing zivid.

    pip install <packages listed in pyproject.toml>
    pip install zivid

### Installing from source

    git clone <zivid-python clone URL>
    cd zivid-python
    pip install .

The above `pip install .` command may give permission issues on some Windows machines. If so, try the following instead:

    python continuous-integration/windows/create_binary_distribution.py
    pip install ./dist/*.whl

You may want to build Zivid Python against a different (but compatible) version of Zivid SDK. An example would be if Zivid SDK 2.1 was released but the official
Zivid Python still formally only supports SDK 2.0. Since all the features of the 2.0 API exist in the 2.1 API, Zivid Python can still be built with the new SDK
(but without wrapping the latest features). In order to achieve this, edit `setup.py` to target the new SDK version before doing `pip install .`. Note that
this option is considered experimental/unofficial.

## Quick Start

### Point cloud capture

To quickly capture a point cloud using default settings, run the following code:
```python
import zivid
app = zivid.Application()
camera = app.connect_camera()
settings = zivid.Settings(
    acquisitions=[zivid.Settings.Acquisition()],
    color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
)
frame = camera.capture_2d_3d(settings)
frame.save("result.zdf")
```

Instead of using the API to define capture settings, it is also possible to load them from YML files that
have been exported from [Zivid Studio][zivid-studio-guide-url] or downloaded from the Zivid Knowledge Base
[settings library][zivid-two-standard-settings-url]. This can be done by providing the filesystem path to
such a file, for example:

```python
settings = Settings.load("ZividTwo_Settings_2xHDR_Normal.yml")
frame = camera.capture_2d_3d(settings)
```

### Point cloud data access

Data can easily be accessed in the form of Numpy arrays:

```python
import zivid
app = zivid.Application()
camera = app.connect_camera()
settings = zivid.Settings(
    acquisitions=[zivid.Settings.Acquisition()],
    color=zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()]),
)
frame = camera.capture_2d_3d(settings)
xyz = frame.point_cloud().copy_data("xyz") # Get point coordinates as [Height,Width,3] float array
rgba = frame.point_cloud().copy_data("rgba") # Get point colors as [Height,Width,4] uint8 array
bgra = frame.point_cloud().copy_data("bgra") # Get point colors as [Height,Width,4] uint8 array
```

### Capture Assistant

Instead of manually adjusting settings, the Capture Assistant may be used to find the optimal settings for your scene:

```python
import zivid
app = zivid.Application()
camera = app.connect_camera()
capture_assistant_params = zivid.capture_assistant.SuggestSettingsParameters()
settings = zivid.capture_assistant.suggest_settings(camera, capture_assistant_params)
frame = camera.capture_2d_3d(settings)
frame.save("result.zdf")
```

### Using camera emulation

If you do not have a camera, you can use the `FileCameraZivid2M70.zfc` file in the [Sample Data][zivid-download-sampledata-url] to emulate a camera.

```python
import zivid
app = zivid.Application()
camera = app.create_file_camera("path/to/FileCameraZivid2M70.zfc")
settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
frame = camera.capture_3d(settings)
frame.save("result.zdf")
```

## Examples

Basic example programs can be found in the [samples](samples) directory.
Many more advanced example programs may be found in the separate
[zivid-python-samples](https://github.com/zivid/zivid-python-samples) repository.

## Versioning

This python module is using [PEP 440](https://www.python.org/dev/peps/pep-0440) for versioning. The features available in the module depends on the Zivid SDK version used when building the module. When updating this Python package it is *recommended* to also update to the latest [Zivid SDK][zivid-software-url]. Refer to the [Test Matrix](#test-matrix) for supported version.

The version number of the Zivid Python module consists of six numbers. The three first numbers of the version is the [semantic version](https://semver.org/) of the code in this repository. The last three numbers is the version of the underlying Zivid SDK library used by the Python module.

To check which version of zivid-python that corresponds to a specific version of Zivid SDK, check out [zivid-python-releases-url] or run `pip index versions zivid`.

### Version breakdown

                                        Zivid SDK version = 1.4.1 (semantic version)
                                        v v v
    Zivid Python module version = 1.0.0.1.4.1
                                  ^ ^ ^
                                  Wrapper code version = 1.0.0 (semantic version)

### PyPI

When installing using PIP it is possible to specify the required version. This can be useful if upgrading Zivid SDK is not desired, but you want to update Zivid Python.

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

This project is licensed, see the [LICENSE](LICENSE) file for details. The licenses of dependencies are listed [here](./licenses-dependencies).

## Support

Please visit [Zivid Knowledge Base][zivid-knowledge-base-url] for general information on using Zivid 3D cameras. If you cannot find a solution to your issue, please contact customersuccess@zivid.com.

## Test matrix

| Operating System  | Python version                        |
|:------------------|:--------------------------------------|
| Ubuntu 24.10      | 3.12                                  |
| Ubuntu 24.04      | 3.12                                  |
| Ubuntu 22.04      | 3.10                                  |
| Ubuntu 20.04      | 3.8                                   |
| Fedora 40         | 3.12                                  |
| Fedora 41         | 3.13                                  |
| Windows 10        | 3.7, 3.8, 3.9, 3.10, 3.11, 3.12, 3.13 |

[header-image]: https://www.zivid.com/hubfs/softwarefiles/images/zivid-generic-github-header.png
[ci-badge]: https://img.shields.io/github/actions/workflow/status/zivid/zivid-python/main.yml?branch=master
[ci-url]: https://github.com/zivid/zivid-python/actions?query=workflow%3A%22Main+CI+workflow%22+branch%3Amaster
[pypi-badge]: https://img.shields.io/pypi/v/zivid.svg
[pypi-url]: https://pypi.org/project/zivid

[zivid-knowledge-base-url]: http://support.zivid.com
[zivid-software-installation-url]: https://support.zivid.com/latest/getting-started/software-installation.html
[zivid-download-software-url]: https://www.zivid.com/downloads
[zivid-download-sampledata-url]: https://support.zivid.com/en/latest/api-reference/samples/sample-data.html
[zivid-software-url]: http://www.zivid.com/software
[zivid-python-releases-url]: https://pypi.org/project/zivid/#history
[zivid-studio-guide-url]: https://support.zivid.com/en/latest/getting-started/studio-guide.html
[zivid-two-standard-settings-url]: https://support.zivid.com/en/latest/reference-articles/standard-acquisition-settings-zivid-two.html
