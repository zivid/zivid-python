"""Contains the UnorganizedPointCloud class."""

import _zivid
import numpy


class UnorganizedPointCloud:
    """Point cloud with x, y, z, RGB color and SNR laid out as a linear list of only valid points.

    An instance of this class is a handle to a point cloud stored on the compute device memory.
    This class provides several methods to copy point cloud data from the compute device
    memory to host (CPU) system memory (RAM).

    This point cloud contains only valid points, meaning that the XYZ values are never NaN.
    """

    def __init__(self, impl=None):
        """Create an empty point cloud.

        This constructor creates a point cloud with size zero, i.e. no points. An empty point cloud
        can be useful for combining points from several other points clouds.

        The argument impl is only used internally, and should not be set by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if impl is None:
            self.__impl = _zivid.UnorganizedPointCloud()
        else:
            if not isinstance(impl, _zivid.UnorganizedPointCloud):
                raise TypeError(
                    "Unsupported type for argument impl. Got {}, expected {}".format(
                        type(impl), _zivid.UnorganizedPointCloud
                    )
                )
            self.__impl = impl

    @property
    def size(self):
        """Get the size of the point cloud (number of points)."""
        return self.__impl.size()

    def extended(self, other):
        """Create a new point cloud containing the combined data of this point cloud and another.

        Args:
            other: The other UnorganizedPointCloud to copy data from

        Returns:
            A new UnorganizedPointCloud containing the combined data
        """
        if not isinstance(other, UnorganizedPointCloud):
            raise TypeError(
                "Unsupported type for argument other. Got {}, expected {}".format(type(other), UnorganizedPointCloud)
            )
        return UnorganizedPointCloud(self.__impl.extended(other.__impl))  # pylint: disable=protected-access

    def extend(self, other):
        """Extend this point cloud in-place by adding the points from another point cloud.

        Args:
            other: The other UnorganizedPointCloud to copy data from

        Returns:
            Reference to the same UnorganizedPointCloud instance (for chaining calls)
        """
        if not isinstance(other, UnorganizedPointCloud):
            raise TypeError(
                "Unsupported type for argument other. Got {}, expected {}".format(type(other), UnorganizedPointCloud)
            )
        self.__impl.extend(other.__impl)  # pylint: disable=protected-access
        return self

    def voxel_downsampled(self, voxel_size, min_points_per_voxel):
        """Create a new point cloud that is a voxel downsampling of this point cloud.

        Voxel downsampling subdivides 3D space into a grid of cubic voxels with a given size. If a given voxel
        contains a number of points at or above the given limit, all those source points are replaced with a
        single new point at the SNR-weighted average position.

        Args:
            voxel_size: The size of the voxel cubes (must be greater than 0.0)
            min_points_per_voxel: The minimum number of points required to fill a voxel (must be 1 or greater)

        Returns:
            A new UnorganizedPointCloud containing the voxel downsampled data
        """
        return UnorganizedPointCloud(self.__impl.voxel_downsampled(voxel_size, min_points_per_voxel))

    def transform(self, matrix):
        """Transform the point cloud in-place by a 4x4 transformation matrix.

        The transform matrix must be affine, i.e., the last row of the matrix should be [0, 0, 0, 1].

        Args:
            matrix: A 4x4 numpy arrays of floats

        Returns:
            Reference to the same UnorganizedPointCloud instance (for chaining calls)
        """
        self.__impl.transform(matrix)
        return self

    def transformed(self, matrix):
        """Get a transformed copy of the point cloud.

        This method is identical to "transform", except the transformed point cloud is
        returned as a new UnorganizedPointCloud instance. The current point cloud is not modified.

        Args:
            matrix: A 4x4 numpy arrays of floats

        Returns:
            A new UnorganizedPointCloud instance
        """
        return UnorganizedPointCloud(self.__impl.transformed(matrix))

    def center(self):
        """Translate the point cloud in-place so that its centroid lands at the origin (0,0,0)."""
        self.__impl.center()
        return self

    def centroid(self):
        """Get the centroid of the point cloud, i.e. average of all XYZ point positions.

        Returns:
            The XYZ centroid as a numpy array, or None if the point cloud is empty.
        """
        return self.__impl.centroid()

    def paint_uniform_color(self, color):
        """Set point cloud colors in-place according to the given value.

        Args:
            color: The RGBA value used to color all points as a list of 8-bit integers List[int[4]] or Numpy array of\
            shape (4,) or (1,4) with dtype=np.uint8
        """
        self.__impl.paint_uniform_color(color)
        return self

    def painted_uniform_color(self, color):
        """Create a clone of this point cloud with all points colored according to the given value.

        Args:
            color: The RGBA value used to color all points as a list of 8-bit integers List[int[4]] or Numpy array of\
            shape (4,) or (1,4) with dtype=np.uint8
        """
        return UnorganizedPointCloud(self.__impl.painted_uniform_color(color))

    def copy_data(self, data_format):
        """Copy point cloud data from GPU to numpy array.

        Supported data formats:
        xyz:            ndarray(Size,3) of float
        rgba:           ndarray(Size,4) of uint8
        bgra:           ndarray(Size,4) of uint8
        rgba_srgb:      ndarray(Size,4) of uint8
        bgra_srgb:      ndarray(Size,4) of uint8
        snr:            ndarray(Size)   of float

        Args:
            data_format: A string specifying the data to be copied

        Returns:
            A numpy array with the requested data.

        Raises:
            ValueError: if the requested data format does not exist
        """
        self.__impl.assert_not_released()

        data_formats = {
            "xyz": _zivid.Array1DPointXYZ,
            "xyzw": _zivid.Array1DPointXYZW,
            "rgba": _zivid.Array1DColorRGBA,
            "bgra": _zivid.Array1DColorBGRA,
            "rgba_srgb": _zivid.Array1DColorRGBA_SRGB,
            "bgra_srgb": _zivid.Array1DColorBGRA_SRGB,
            "snr": _zivid.Array1DSNR,
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

    def clone(self):
        """Get a clone of the point cloud.

        The clone will include a copy of all the point cloud data on the compute device memory. This means that the
        returned point cloud will not be affected by subsequent modifications (such as transform or downsample) on the
        original point cloud.

        This function incurs a performance cost due to the copying of the compute device memory. When performance is
        important we recommend to avoid using this method, and instead modify the existing point cloud.

        This method is equivalent to calling `copy.deepcopy` on the point cloud. You can obtain a shallow copy that does
        not copy the underlying data by using `copy.copy` on the point cloud instead.

        Returns:
            A new UnorganizedPointCloud instance
        """
        return UnorganizedPointCloud(self.__impl.clone())

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

    def __copy__(self):
        return UnorganizedPointCloud(self.__impl.__copy__())

    def __deepcopy__(self, memodict):
        return UnorganizedPointCloud(self.__impl.__deepcopy__(memodict))
