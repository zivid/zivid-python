"""This file imports used classes, modules and packages."""

import ctypes
import importlib
import platform
import sys
from pathlib import Path

if platform.system() == "Windows" and sys.version_info.major == 3 and sys.version_info.minor >= 8:
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
    if len(pyd_files) != 1:
        raise ImportError(f"Expected exactly one _zivid*.pyd file in {package_dir}, found {len(pyd_files)} files.")
    ctypes.WinDLL(str(pyd_files[0]), winmode=0)

try:
    from _zivid._zivid import (  # pylint: disable=import-error,no-name-in-module
        Application,
        Array1DColorBGRA,
        Array1DColorBGRA_SRGB,
        Array1DColorRGBA,
        Array1DColorRGBA_SRGB,
        Array1DPointXYZ,
        Array1DSNR,
        Array2DColorBGRA,
        Array2DColorBGRA_SRGB,
        Array2DColorRGBA,
        Array2DColorRGBA_SRGB,
        Array2DNormalXYZ,
        Array2DPointXYZ,
        Array2DPointXYZColorBGRA,
        Array2DPointXYZColorBGRA_SRGB,
        Array2DPointXYZColorRGBA,
        Array2DPointXYZColorRGBA_SRGB,
        Array2DPointXYZW,
        Array2DPointZ,
        Array2DSNR,
        Camera,
        CameraInfo,
        CameraIntrinsics,
        CameraState,
        Frame,
        Frame2D,
        FrameInfo,
        ImageBGRA,
        ImageBGRA_SRGB,
        ImageRGBA,
        ImageRGBA_SRGB,
        Matrix4x4,
        NetworkConfiguration,
        PixelMapping,
        PointCloud,
        ProjectedImage,
        SceneConditions,
        Settings,
        Settings2D,
        UnorganizedPointCloud,
        __version__,
        calibration,
        capture_assistant,
        data_model,
        firmware,
        infield_correction,
        point_cloud_export,
        presets,
        projection,
        toolbox,
        version,
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
