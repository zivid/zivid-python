"""Contains the PointCloud class."""
import numpy

import _zivid


class PointCloud:
    """A point cloud."""

    def __init__(self, internal_point_cloud):
        """Create a point cloud from an internal point cloud.

        Args:
            internal_point_cloud: a internal point cloud

        """
        if not isinstance(internal_point_cloud, _zivid.PointCloud):
            raise ValueError(
                "Unsupported type for argument internal_point_cloud. Got {}, expected {}".format(
                    type(internal_point_cloud), type(_zivid.PointCloud)
                )
            )
        self.__impl = internal_point_cloud

    def to_array(self):
        """Convert point cloud to numpy array.

        Returns:
            a numpy array

        """
        self.__impl.assert_not_released()
        return numpy.array(self.__impl)

    @property
    def height(self):
        """Return height (number of rows) of point cloud.

        Returns:
            a positive integer

        """
        return self.__impl.height()

    @property
    def width(self):
        """Return width (number of columns) of point cloud.

        Returns:
            a positive integer

        """
        return self.__impl.width()

    def release(self):
        """Release the underlying resources."""
        self.__impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
