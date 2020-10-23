"""Contains the PointCloud class."""
import numpy

import _zivid


class PointCloud:
    """Point cloud with x, y, z, RGB and color laid out on a 2D grid.

    An instance of this class is a handle to a point cloud stored on the compute device memory.
    Use the method copy_data to copy point cloud data from the compute device to a numpy
    array in host memory. Several formats are available.
    """

    class Downsampling:  # pylint: disable=too-few-public-methods
        """Collection of valid options to PointCloud.downsample()."""

        by2x2 = "by2x2"
        by3x3 = "by3x3"
        by4x4 = "by4x4"

        _valid_values = {
            "by2x2": _zivid.PointCloud.Downsampling.by2x2,
            "by3x3": _zivid.PointCloud.Downsampling.by3x3,
            "by4x4": _zivid.PointCloud.Downsampling.by4x4,
        }

        @classmethod
        def valid_values(cls):
            """Get list of allowed values.

            Returns:
                List of strings
            """
            return list(cls._valid_values.keys())

    def __init__(self, impl):
        """Initialize PointCloud wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.PointCloud):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.PointCloud)
                )
            )
        self.__impl = impl

    def copy_data(self, data_format):
        """Copy point cloud data from GPU to numpy array.

        Supported data formats:
        xyz:        ndarray(Height,Width,3) of float
        xyzw:       ndarray(Height,Width,4) of float
        z:          ndarray(Height,Width)   of float
        rgba:       ndarray(Height,Width,4) of uint8
        snr:        ndarray(Height,Width)   of float
        xyzrgba:    ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])

        Args:
            data_format: A string specifying the data to be copied

        Returns:
            A numpy array with the requested data.

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
            matrix: A 4x4 numpy arrays of floats

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        self.__impl.transform(matrix)
        return self

    def downsample(self, downsampling):
        """Downsample the point cloud in-place.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        internal_downsampling = PointCloud.Downsampling._valid_values[  # pylint: disable=protected-access
            downsampling
        ]
        self.__impl.downsample(internal_downsampling)
        return self

    def downsampled(self, downsampling):
        """Get a downsampled copy of the point cloud.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            A new PointCloud instance
        """
        internal_downsampling = PointCloud.Downsampling._valid_values[  # pylint: disable=protected-access
            downsampling
        ]
        return PointCloud(self.__impl.downsampled(internal_downsampling))

    @property
    def height(self):
        """Get the height of the point cloud (number of rows).

        Returns:
            A positive integer
        """
        return self.__impl.height()

    @property
    def width(self):
        """Get the width of the point cloud (number of columns).

        Returns:
            A positive integer
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
