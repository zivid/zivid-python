"""Contains FrameInfo class."""
import _zivid


class FrameInfo:  # pylint: disable=too-few-public-methods
    """Various information for a frame."""

    def __init__(self, timestamp=_zivid.FrameInfo.TimeStamp().value):
        """Initialize frame info.

        Args:
            timestamp: The time of frame capture

        """
        self.timestamp = timestamp
