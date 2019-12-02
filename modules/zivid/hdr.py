"""Contains a global HDR method."""
import _zivid
from zivid.frame import Frame
import zivid._settings_converter as _settings_converter


def combine_frames(frames):
    """Combine frames acquired with different camera settings.

    This function will return a high dynamic range frame based on a supplied
        set of frames captured with different camera settings

    Args:
        frames: a sequence of frame objects

    Returns:
        a frame instance

    """
    return Frame(
        _zivid.hdr.combine_frames(
            [frame._Frame__impl for frame in frames]  # pylint: disable=protected-access
        )
    )


def capture(camera, settings_list):
    """A convenience function to capture and merge frames.

    Args:
        camera: a reference to camera instance
        settings_list: a list of camera settings

    Returns:
        a single frame (when single setting given) or high dynamic range frame (when multiple settings given)

    """
    return Frame(_zivid.hdr.capture(camera._Camera__impl, \
                                    [_settings_converter.to_internal_settings(settings) for settings in settings_list]))  # pylint: disable=protected-access
