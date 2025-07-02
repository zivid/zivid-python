# Zivid Python

Zivid Python is the official Python package for Zivid 3D cameras. Read more about Zivid
at [zivid.com](https://www.zivid.com/).

[![Build Status][ci-badge]][ci-url] [![PyPI Package][pypi-badge]][pypi-url]
![Zivid Image][header-image]

---

*Contents:* **[Installation](#installation)** | **[Quick Start](#quick-start)** | **[Examples](#examples)** |
**[Versioning](#versioning)** | **[License](#license)** | **[Support](#support)** | **[Test Matrix](#test-matrix)**

---

## Installation

### Dependencies

* [Python](https://www.python.org/) version 3.7 or higher
* [Zivid SDK][zivid-download-software-url] version 2.16.0 (see [here][zivid-software-installation-url] for help)
* [Compiler](doc/CompilerInstallation.md) with C++17 support

*Ubuntu users must install Python headers (`apt install python3-dev`) in addition to the regular `python3` package.*

*Windows users also needs to make sure that the Zivid SDK installation folder is in system `PATH` before using the
package, not only the terminal PATH variable. The default install location that should be added to system `PATH`
is `C:\Program Files\Zivid\bin`.*

### Installing official version from PyPI using PIP

After having installed the latest Zivid SDK, the easiest way to install Zivid Python is to use PIP to fetch the latest
official version from PyPI:

```shell
pip install zivid
```

Note:

> If you don't use the latest Zivid SDK version you need to manually specify the version. See [Versioning](#versioning).

Installation may take some time since the `setup.py` script will download additional dependencies and compile C++ source
code in the background.

On some systems Python 3 `pip` is called `pip3`. In this guide we assume it is called `pip`. When using PIP version 19
or higher build dependencies are handled automatically.

#### Old PIP

If you are using a version of PIP older than version 19 please manually install the dependencies listed
in [pyproject.toml](pyproject.toml) before installing zivid.

```shell
pip install <packages listed in pyproject.toml>
pip install zivid
```

### Installing from source

```shell
git clone <zivid-python clone URL>
cd zivid-python
pip install .
```

The above `pip install .` command may give permission issues on some Windows machines. If so, try the following instead:

```shell
python continuous-integration/windows/create_binary_distribution.py
pip install ./dist/*.whl
```

You may want to build Zivid Python against a different (but compatible) version of Zivid SDK. An example would be if
Zivid SDK 2.1 was released but the official
Zivid Python still formally only supports SDK 2.0. Since all the features of the 2.0 API exist in the 2.1 API, Zivid
Python can still be built with the new SDK
(but without wrapping the latest features). In order to achieve this, edit `sdk_version.json` to target the new SDK
version before doing `pip install .`. Note that this option is considered experimental/unofficial.

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
xyz = frame.point_cloud().copy_data("xyz")  # Get point coordinates as [Height,Width,3] float array
rgba = frame.point_cloud().copy_data("rgba")  # Get point colors as [Height,Width,4] uint8 array
bgra = frame.point_cloud().copy_data("bgra")  # Get point colors as [Height,Width,4] uint8 array
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

If you do not have a camera, you can use the `FileCameraZivid2M70.zfc` file in
the [Sample Data][zivid-download-sampledata-url] to emulate a camera.

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

This python module is released with the same version number as the Zivid SDK that it supports. The Zivid SDK is using
[semantic versioning](https://semver.org/) with major, minor, and patch versions.

If a patch is released to fix an issue with this python module separately from a Zivid SDK patch release, then a fourth
number signifying the patch version of the python module will be added to the end:

### Version breakdown

```text
                              Zivid SDK version = 2.16.0
                              v vv v
Zivid Python module version = 2.16.0.1
                                     ^
                                     Zivid Python module patch version (omitted if 0)
```

> [!NOTE]
> **Versioning Prior to 2.16.0**
>
> Before version 2.16.0, this python module used a similar but different versioning scheme with six numbers. In this
> scheme, the first three numbers specified the semantic version of the python module while the next three numbers
> specified the semantic version of the supported Zivid SDK. In some early versions of the python module, these semantic
> versions could be different, but eventually they were synced up, and from version 2.16.0 the versioning system was
> simplified as explained above.

### PyPI

When installing using PIP it is possible to specify the required version. This can be useful if using an older version
of the Zivid SDK.

To see the complete list of released versions of this python module, see [zivid-python-releases-url] or run
`pip index versions zivid`.

Note that as explained above, the versioning system was simplified starting with version 2.16.0 such that the Zivid
Python is the same as the supported Zivid SDK version. Older releases used a different versioning scheme, which is also
explained above.

#### Install latest version of Zivid Python using latest available version of Zivid SDK

```shell
pip install zivid
```

Note: The installation may fail if the latest available version of Zivid SDK is not installed on the system. See
[Installation](#installation).

#### Install a specific version of Zivid Python

```shell
pip install zivid==2.16.0
```

This requires Zivid SDK version 2.16.0 to be installed on the system.

#### Using the old versioning scheme, install version 2.6.0 of the Zivid Python wrapper supporting Zivid SDK 2.7.0

```shell
pip install zivid==2.6.0.2.7.0
```

This requires Zivid SDK version 2.7.0 to be installed on the system.

## License

This project is licensed, see the [LICENSE](LICENSE) file for details. The licenses of dependencies are
listed [here](./licenses-dependencies).

## Support

Please visit [Zivid Knowledge Base][zivid-knowledge-base-url] for general information on using Zivid 3D cameras. If you
cannot find a solution to your issue, please contact <customersuccess@zivid.com>.

## Test matrix

The test matrix shows which Python versions and operating systems are tested in CI.
Click [here](continuous-integration/TestMatrix.md) to go to the test matrix.

[header-image]: https://www.zivid.com/hubfs/softwarefiles/images/zivid-generic-github-header.png

[ci-badge]: https://img.shields.io/github/actions/workflow/status/zivid/zivid-python/main.yml?branch=master

[ci-url]: https://github.com/zivid/zivid-python/actions?query=workflow%3A%22Main+CI+workflow%22+branch%3Amaster

[pypi-badge]: https://img.shields.io/pypi/v/zivid.svg

[pypi-url]: https://pypi.org/project/zivid

[zivid-knowledge-base-url]: http://support.zivid.com

[zivid-software-installation-url]: https://support.zivid.com/latest/getting-started/software-installation.html

[zivid-download-software-url]: https://www.zivid.com/downloads

[zivid-download-sampledata-url]: https://support.zivid.com/en/latest/api-reference/samples/sample-data.html

[zivid-python-releases-url]: https://pypi.org/project/zivid/#history

[zivid-studio-guide-url]: https://support.zivid.com/en/latest/getting-started/studio-guide.html

[zivid-two-standard-settings-url]: https://support.zivid.com/en/latest/reference-articles/standard-acquisition-settings-zivid-two.html
