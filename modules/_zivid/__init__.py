"""This file imports used classes, modules and packages."""

import ctypes
import importlib
import platform
import sys
from pathlib import Path

if (
    platform.system() == "Windows"
    and sys.version_info.major == 3
    and sys.version_info.minor >= 8
):
    # Starting with Python 3.8, the .dll search mechanism has changed.
    # WinDLL has anew argument "winmode",
    # https://docs.python.org/3.8/library/ctypes.html
    # and it turns out that we MUST import the pybind11 generated module
    # with "winmode=0". After doing this, the following import statement
    # picks this up since it's already imported, and things work as intended.
    #
    # The winmode parameter is used on Windows to specify how the library is
    # loaded (since mode is ignored). It takes any value that is valid for the
    # Win32 API LoadLibraryEx flags parameter. When omitted, the default is to
    # use the flags that result in the most secure DLL load to avoiding issues
    # such as DLL hijacking. Passing winmode=0 passes 0 as dwFlags to
    # LoadLibraryExA:
    # https://docs.microsoft.com/en-us/windows/win32/api/libloaderapi/nf-libloaderapi-loadlibraryexa

    package_dir = Path(importlib.util.find_spec("_zivid").origin).parent
    pyd_files = list(package_dir.glob("_zivid*.pyd"))
    assert len(pyd_files) == 1
    ctypes.WinDLL(str(pyd_files[0]), winmode=0)

try:
    from _zivid._zivid import (  # pylint: disable=import-error,no-name-in-module
        __version__,
        Application,
        Array2DColorRGBA,
        Array2DPointXYZ,
        Array2DPointXYZColorRGBA,
        Array2DPointXYZW,
        Array2DPointZ,
        Array2DSNR,
        Camera,
        CameraState,
        firmware,
        calibration,
        capture_assistant,
        Frame,
        FrameInfo,
        PointCloud,
        Settings,
        version,
        Settings2D,
        Frame2D,
        ImageRGBA,
        CameraInfo,
    )
except ImportError as ex:

    def __missing_sdk_error_message():
        error_message = """Failed to import the Zivid Python C-module, please verify that:
 - Zivid SDK is installed
 - Zivid SDK version is matching the SDK version part of the Zivid Python version """
        if platform.system() != "Windows":
            return error_message
        return (
            error_message
            + """
 - Zivid SDK libraries location is in system PATH"""
        )

    raise ImportError(__missing_sdk_error_message()) from ex
