"""Module containing implementation of feature point detection functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""
import _zivid
from zivid._calibration.pose import Pose


class DetectionResult:
    """Class representing detected feature points."""

    def __init__(self, impl):
        """Initialize DetectionResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.DetectionResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.calibration.DetectionResult)
                )
            )

        self.__impl = impl

    def valid(self):
        """Check validity of DetectionResult.

        Returns:
            True if DetectionResult is valid
        """
        return self.__impl.valid()

    def centroid(self):
        """Get the centroid of the detected feature points.

        Will throw an exception if the DetectionResult is not valid.

        Returns:
            A 1D array containing the X, Y and Z coordinates of the centroid
        """
        return self.__impl.centroid()

    def pose(self):
        """Get position and orientation of the top left detected corner in camera-space.

        Pose calculation works for official Zivid calibration boards only.
        An exception will be thrown if valid() is false or if the board is not supported.

        Returns:
            The Pose of the top left corner (4x4 transformation matrix)
        """
        return Pose(self.__impl.pose().to_matrix())

    def __bool__(self):
        return bool(self.__impl)

    def __str__(self):
        return str(self.__impl)


def detect_feature_points(point_cloud):
    """Detect feature points from a calibration object in a point cloud.

    Args:
        point_cloud: PointCloud containing a calibration object

    Returns:
        A DetectionResult instance
    """

    return DetectionResult(
        _zivid.calibration.detect_feature_points(
            point_cloud._PointCloud__impl  # pylint: disable=protected-access
        )
    )
