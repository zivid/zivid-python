"""Contains the PointCloud class."""

import numpy

import _zivid
from zivid.image import Image


class PointCloud:
    """Point cloud with x, y, z, RGB and color laid out on a 2D grid.

    An instance of this class is a handle to a point cloud stored on the compute device memory.
    Use the method copy_data to copy point cloud data from the compute device to a numpy
    array in host memory. Several formats are available.

    If the point cloud is the result of a 2D+3D capture, the RGB colors will be set from the captured 2D color image.
    If different pixel sampling (resolution) settings for 2D and 3D were used, or if the point cloud is upsampled or
    downsampled, then the RGB colors will be resampled to correspond 1:1 with the 3D point cloud resolution.
    To get the original resolution 2D color image from the 2D+3D capture, see the frame_2d method of the Frame class.

    If the point cloud is the result of a 3D-only capture, the RGB colors will be set to a uniform default color.
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
                    type(impl), _zivid.PointCloud
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
        bgra:       ndarray(Height,Width,4) of uint8
        srgb:       ndarray(Height,Width,4) of uint8
        normals:    ndarray(Height,Width,3) of float
        snr:        ndarray(Height,Width)   of float
        xyzrgba:    ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])
        xyzbgra:    ndarray(Height,Width)   of composite dtype (accessed with e.g. arr["x"])

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
            "bgra": _zivid.Array2DColorBGRA,
            "srgb": _zivid.Array2DColorSRGB,
            "normals": _zivid.Array2DNormalXYZ,
            "snr": _zivid.Array2DSNR,
            "xyzrgba": _zivid.Array2DPointXYZColorRGBA,
            "xyzbgra": _zivid.Array2DPointXYZColorBGRA,
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

    def copy_image(self, data_format):
        """Copy the point cloud colors as 8-bit image in input format.

        Supported data formats:
        rgba:       Image(Height,Width,4) of uint8
        bgra:       Image(Height,Width,4) of uint8
        srgb:       Image(Height,Width,4) of uint8

        Args:
            data_format: A string specifying the image data format

        Returns:
            An image instance containing color data

        Raises:
            ValueError: if the requested data format does not exist
        """
        self.__impl.assert_not_released()

        supported_color_formats = ["rgba", "bgra", "srgb"]

        if data_format == "rgba":
            return Image(self.__impl.copy_image_rgba())
        if data_format == "bgra":
            return Image(self.__impl.copy_image_bgra())
        if data_format == "srgb":
            return Image(self.__impl.copy_image_srgb())
        raise ValueError(
            "Unsupported color format: {data_format}. Supported formats: {all_formats}".format(
                data_format=data_format, all_formats=supported_color_formats
            )
        )

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

        Downsampling is used to reduce the number of points in the point cloud. Downsampling is performed
        by combining a 2x2, 3x3 or 4x4 region of pixels in the original point cloud to one pixel in the
        new point cloud. A downsampling factor of 2x2 will reduce width and height each to half, and thus
        the overall number of points to 1/4. 3x3 downsampling reduces width and height each to 1/3, and
        the overall number of points to 1/9, and so on.

        X, Y and Z coordinates are downsampled by computing the SNR^2 weighted average of each point in
        the corresponding NxN region in the original point cloud, ignoring invalid (NaN) points. Color is
        downsampled by computing the average value for each color channel in the NxN region. SNR value is
        downsampled by computing the square root of the sum of SNR^2 of each valid (non-NaN) point in the
        NxN region. If all points in the NxN region are invalid (NaN), the downsampled SNR is set to the
        max SNR in the region.

        As an alternative to using this method, downsampling may also be specified up-front when capturing
        by using Settings/Processing/Resampling.

        Downsampling is performed on the compute device. The point cloud is modified in-place. Use
        "downsampled" if you want to downsample to a new PointCloud instance. Downsampling
        can be repeated multiple times to further reduce the size of the point cloud, if desired.

        Note that the width or height of the point cloud is not required to divide evenly by the
        downsampling factor (2, 3 or 4). The new width and height equals the original width and height
        divided by the downsampling factor, rounded down. In this case the remaining columns at the right
        and/or rows at the bottom of the original point cloud are ignored.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            Reference to the same PointCloud instance (for chaining calls)
        """
        internal_downsampling = (
            PointCloud.Downsampling._valid_values[  # pylint: disable=protected-access
                downsampling
            ]
        )
        self.__impl.downsample(internal_downsampling)
        return self

    def downsampled(self, downsampling):
        """Get a downsampled copy of the point cloud.

        This method is identical to "downsample", except the downsampled point cloud is
        returned as a new PointCloud instance. The current point cloud is not modified.

        Args:
            downsampling: One of the strings in PointCloud.Downsample.valid_values()

        Returns:
            A new PointCloud instance
        """
        internal_downsampling = (
            PointCloud.Downsampling._valid_values[  # pylint: disable=protected-access
                downsampling
            ]
        )
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
