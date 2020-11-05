"""Contains functions for checking and updating camera firmware."""
import _zivid


def update(camera, progress_callback=None):
    """Update camera firmware.

    If the current API requires a different firmware than what is present on the camera,
        the firmware will be updated to this version.
    The function throws if the camera is connected, or if the camera is already up to date.
    Call is_up_to_date() first to check if the camera is up to date.

    Args:
        camera: The camera to be updated.
        progress_callback: A callback function to track progress of update.
            The callable is taking a float and a string as progress and description respectively.
    """
    if progress_callback is None:
        _zivid.firmware.update(camera._Camera__impl)  # pylint: disable=protected-access
    else:
        _zivid.firmware.update(
            camera._Camera__impl,  # pylint: disable=protected-access
            progress_callback=progress_callback,
        )


def is_up_to_date(camera):
    """Check if the firmware on the camera is of the version that is required by the API.

    Args:
        camera: The camera to check the firmware of (must be in disconnected state)

    Returns:
        A bool that is True if the firmware is up to date
    """
    return _zivid.firmware.is_up_to_date(
        camera._Camera__impl  # pylint: disable=protected-access
    )
