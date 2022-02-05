"""Module for experimental calibration features. This API may change in the future."""

import _zivid
from zivid.camera_intrinsics import _to_camera_intrinsics


def intrinsics(camera):
    """Get intrinsic parameters of a given camera.

    Args:
        camera: A Camera instance

    Returns:
        A CameraIntrinsics instance
    """
    return _to_camera_intrinsics(
        _zivid.calibration.intrinsics(
            camera._Camera__impl  # pylint: disable=protected-access
        )
    )


def estimate_intrinsics(frame):
    """Estimate camera intrinsics for a given frame.

    This function is for advanced use cases. Otherwise, use intrinsics(camera).

    Args:
        frame: A Frame instance

    Returns:
        A CameraIntrinsics instance
    """
    return _to_camera_intrinsics(
        _zivid.calibration.estimate_intrinsics(
            frame._Frame__impl  # pylint: disable=protected-access
        )
    )
