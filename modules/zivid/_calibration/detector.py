"""Module containing implementation of feature point detection functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""

import _zivid
from zivid._calibration.pose import Pose


class DetectionResult:
    """Class representing detected feature points from a calibration board."""

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
                    type(impl), _zivid.calibration.DetectionResult
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


class MarkerShape:
    """Holds physical (3D) and image (2D) properties of a detected fiducial marker"""

    def __init__(self, impl):
        """Initialize MarkerShape wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.MarkerShape):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.calibration.MarkerShape
                )
            )

        self.__impl = impl

    def corners_in_pixel_coordinates(self):
        """Get 2D image coordinates of the corners of the detected marker.

        Returns:
            List of four numpy ndarrays
        """
        return self.__impl.corners_in_pixel_coordinates()

    def corners_in_camera_coordinates(self):
        """Get 3D spatial coordinates of the corners of the detected marker.

        Returns:
            List of four numpy ndarrays
        """
        return self.__impl.corners_in_camera_coordinates()

    def id(self):
        """Get the id of the detected marker.

        Returns:
            Id as int
        """
        return self.__impl.id()

    def pose(self):
        """Get 3D pose of the marker.

        Returns:
            The Pose of the marker center (4x4 transformation matrix)
        """
        return self.__impl.pose()


class DetectionResultFiducialMarkers:
    """Class representing detected fiducial markers"""

    def __init__(self, impl):
        """Initialize DetectionResultFiducialMarkers wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.DetectionResultFiducialMarkers):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.calibration.DetectionResultFiducialMarkers
                )
            )

        self.__impl = impl

    def valid(self):
        """Check validity of DetectionResult.

        Returns:
            True if DetectionResult is valid
        """
        return self.__impl.valid()

    def allowed_marker_ids(self):
        """Get the allowed marker ids this detection result was made with.

        Returns:
            A list of integers, equal to what was passed to the detection function.
        """
        return self.__impl.allowed_marker_ids()

    def detected_markers(self):
        """Get all detected markers.

        Returns:
            A list of MarkerShape instances
        """
        return [MarkerShape(impl) for impl in self.__impl.detected_markers()]

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


def detect_markers(frame, allowed_marker_ids):
    """Detects fiducial markers such as ArUco markers in a frame.

    //TODO(ESKIL): Also let user choose dictionary.

    Returns:
        A DetectionResultFiducialMarkers instance
    """

    return DetectionResultFiducialMarkers(
        _zivid.calibration.detect_markers(
            frame._Frame__impl,  # pylint: disable=protected-access
            allowed_marker_ids,
        )
    )
