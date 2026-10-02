"""Contains the FrameFileType enum and the read_frame_file_type function."""

from enum import Enum

import _zivid


class FrameFileType(Enum):
    """The type of frame stored in a .zdf file.

    A .zdf file holds either a 3D frame, loaded with ``zivid.Frame``, or a 2D frame, loaded with
    ``zivid.Frame2D``.
    """

    frame = 0
    frame_2d = 1

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value


def read_frame_file_type(file_name) -> FrameFileType:
    """Get the type of frame stored in a .zdf file.

    This function reads only the file header and does not load the full frame data. Use it to
    choose between ``zivid.Frame`` and ``zivid.Frame2D`` to load the file.

    Args:
        file_name: Path to the .zdf file, as a string or pathlib.Path

    Returns:
        The type of frame stored in the file
    """
    internal_frame_file_type = _zivid._zividcore.read_frame_file_type(  # pylint: disable=protected-access
        str(file_name)
    )
    return FrameFileType[internal_frame_file_type.name]
