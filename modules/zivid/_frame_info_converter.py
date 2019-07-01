"""Contains functions to convert between frame info and internal frame info."""
from zivid.frame_info import FrameInfo


def to_info(internal_frame_info):
    """Convert internal frame info to frame info.

    Args:
        internal_frame_info: a internal frame info object

    Returns:
        a frame info object

    """
    frame_info = FrameInfo(timestamp=internal_frame_info.timestamp.value)
    return frame_info
