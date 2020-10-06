"""Contains the PointCloud class."""
import numpy

import _zivid


class PointCloud:
    """Point cloud with x, y, z, RGB and color laid out on a 2D grid.

    An instance of this class is a handle to a point cloud stored on the compute device memory.
    This class provides several methods to copy point cloud data from the compute device
    memory to host (CPU) system memory (RAM).
    """

    def __init__(self, impl):  # noqa: D107
        if not isinstance(impl, _zivid.PointCloud):
            raise TypeError(
                "Unsupported type for argument internal_point_cloud. Got {}, expected {}".format(
                    type(impl), type(_zivid.PointCloud)
                )
            )
        self.__impl = impl

    def copy_data(self, data_format):
        """Copy point cloud data from GPU to numpy array.

        Supported data formats:
        xyz:        ndarray(H,W,3) of float
        xyzw:       ndarray(H,W,4) of float
        z:          ndarray(H,W)   of float
        rgba:       ndarray(H,W,4) of uint8
        snr:        ndarray(H,W)   of float
        xyzrgba:    ndarray(H,W)   of composite dtype (accessed with e.g. arr["x"])

        Args:
            data_format: a string specifying the data to be copied

        Returns:
            a numpy array with the requested data

        Raises:
            ValueError: if the requested data format does not exist
        """
        self.__impl.assert_not_released()

        data_formats = {
            "xyz": _zivid.Array2DPointXYZ,
            "xyzw": _zivid.Array2DPointXYZW,
            "z": _zivid.Array2DPointZ,
            "rgba": _zivid.Array2DColorRGBA,
            "snr": _zivid.Array2DSNR,
            "xyzrgba": _zivid.Array2DPointXYZColorRGBA,
        }
        try:
            data_format_class = data_formats[data_format]
        except KeyError as ex:
            raise ValueError(
                "Unsupported data format: {data_format}. Supported formats: {all_formats}".format(
                    data_format=data_format, all_formats=list(data_formats.keys())
                )
            ) from ex
        return numpy.array(data_format_class(self.__impl))

    def transform(self, matrix):
        """Transform the point cloud in-place by a 4x4 transformation matrix.

        The transform matrix must be affine, i.e., the last row of the matrix should be [0, 0, 0, 1].

        Args:
            matrix: a 4x4 numpy arrays of floats
        """
        self.__impl.transform(matrix)

    @property
    def height(self):
        """Get the height of the point cloud (number of rows).

        Returns:
            a positive integer

        """
        return self.__impl.height()

    @property
    def width(self):
        """Get the width of the point cloud (number of columns).

        Returns:
            a positive integer

        """
        return self.__impl.width()

    def release(self):
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
