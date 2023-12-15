"""Module for experimental pixel mapping. This API may change in the future."""

import _zivid


class PixelMapping:
    """Pixel mapping from subsampled to full resolution.

    Required when mapping an index in a subsampled point cloud to e.g. a full resolution 2D image.
    """

    def __init__(self, row_stride=1, col_stride=1, row_offset=0.0, col_offset=0.0):
        self.__impl = _zivid.PixelMapping(
            row_stride, col_stride, row_offset, col_offset
        )

    @property
    def row_stride(self):
        return self.__impl.row_stride()

    @property
    def col_stride(self):
        return self.__impl.col_stride()

    @property
    def row_offset(self):
        return self.__impl.row_offset()

    @property
    def col_offset(self):
        return self.__impl.col_offset()

    def __str__(self):
        return str(self.__impl)
