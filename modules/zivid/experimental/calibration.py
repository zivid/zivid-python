"""Module for experimental calibration features. This API may change in the future."""

import _zivid
from zivid.camera_intrinsics import _to_camera_intrinsics
from zivid.experimental import PixelMapping
from zivid.settings import Settings, _to_internal_settings
from zivid.settings2d import Settings2D, _to_internal_settings2d


def intrinsics(camera, settings=None):
    """Get intrinsic parameters of a given camera and settings (3D or 2D).

    These intrinsic parameters take into account the expected resolution of the point clouds captured
    with the given settings. If settings are not provided, intrinsics appropriate for the camera's
    default 3D capture settings is returned.

    For a 2D+3D capture, the 2D color image and 3D point cloud may have different resolutions, depending
    on the pixel sampling and resampling settings used in the Settings and in the Settings.color. This function
    returns intrinsics applicable for the 3D point cloud resolution. You can call this function with a Settings2D
    instance for the 2D intrinsics.

    Args:
        camera: A Camera instance
        settings: Settings or Settings2D to be used to get correct intrinsics (optional)

    Returns:
        A CameraIntrinsics instance

    Raises:
        TypeError: If settings argument is not Settings or Settings2D
    """
    if settings is None:
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(camera._Camera__impl)  # pylint: disable=protected-access
        )
    if isinstance(settings, Settings):
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(
                camera._Camera__impl,  # pylint: disable=protected-access
                _to_internal_settings(settings),
            )
        )
    if isinstance(settings, Settings2D):
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(
                camera._Camera__impl,  # pylint: disable=protected-access
                _to_internal_settings2d(settings),
            )
        )
    raise TypeError(
        "Unsupported type for argument settings. Got {}, expected Settings or Settings2D.".format(type(settings))
    )


def estimate_intrinsics(frame):
    """Estimate camera intrinsics for a given frame.

    The estimated parameters may be used to project 3D point cloud onto the corresponding 2D image.
    This function is for advanced use cases. Otherwise, use intrinsics(camera).

    For a 2D+3D capture, the 2D color image and 3D point cloud may have different resolutions, depending
    on the pixel sampling and resampling settings used in the Settings and in the Settings.color. This function
    returns intrinsics applicable for the 3D point cloud resolution.

    Args:
        frame: A Frame instance

    Returns:
        A CameraIntrinsics instance
    """
    return _to_camera_intrinsics(
        _zivid.calibration.estimate_intrinsics(frame._Frame__impl)  # pylint: disable=protected-access
    )


def pixel_mapping(camera, settings):
    """Return pixel mapping information given camera and settings.

    When mapping from a subsampled point cloud to a full resolution
    2D image it is important to get the pixel mapping correct. This
    mapping depends on camera model and settings. This function provides
    the correct parameters to map the 2D coordinates in a point cloud
    captured using `settings` to the full resolution of the camera.

    Args:
        camera: Reference to camera instance.
        settings: Reference to settings instance.

    Returns:
        A PixelMapping instance.
    """
    pixel_mapping_handle = _zivid.calibration.pixel_mapping(
        camera._Camera__impl,  # pylint: disable=protected-access
        _to_internal_settings(settings),
    )
    return PixelMapping(
        pixel_mapping_handle.row_stride(),
        pixel_mapping_handle.col_stride(),
        pixel_mapping_handle.row_offset(),
        pixel_mapping_handle.col_offset(),
    )
