"""Contains a global HDR method."""
import _zivid
from zivid.frame import Frame


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
