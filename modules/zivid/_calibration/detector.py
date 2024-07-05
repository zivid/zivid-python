"""Module containing implementation of feature point detection functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""

import numpy

import _zivid
from zivid.camera import Camera
from zivid.frame import Frame
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
    """Holds physical (3D) and image (2D) properties of a detected fiducial marker."""

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

    @property
    def corners_in_pixel_coordinates(self):
        """Get 2D image coordinates of the corners of the detected marker.

        Returns:
            Four 2D corner coordinates as a 4x2 numpy array
        """
        return numpy.array(self.__impl.corners_in_pixel_coordinates())

    @property
    def corners_in_camera_coordinates(self):
        """Get 3D spatial coordinates of the corners of the detected marker.

        Returns:
            Four 3D corner coordinates as a 4x3 numpy array
        """
        return numpy.array(self.__impl.corners_in_camera_coordinates())

    @property
    def identifier(self):
        """Get the id of the detected marker.

        Returns:
            Id as int
        """
        return self.__impl.id_()

    @property
    def pose(self):
        """Get 3D pose of the marker.

        The returned pose will be positioned at the center of the marker, and have an orientation such that its z-axis
        points perpendicularly into the face of the marker.

        Returns:
            The Pose of the marker center (4x4 transformation matrix)
        """
        return Pose(self.__impl.pose().to_matrix())


class MarkerDictionary:
    """Holds information about fiducial markers such as ArUco or AprilTag for detection.

    This class's properties describe the different dictionaries available, for example
    aruco4x4_50 describes the ArUco dictionary with 50 markers of size 4x4.

    For more information on ArUco markers see the OpenCV documentation on ArUco markers:
    https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html,

    To get more information about fiducial markers in general, refer to the wikipedia page:
    https://en.wikipedia.org/wiki/Fiducial_marker
    """

    aruco4x4_50 = "aruco4x4_50"
    aruco4x4_100 = "aruco4x4_100"
    aruco4x4_250 = "aruco4x4_250"
    aruco4x4_1000 = "aruco4x4_1000"
    aruco5x5_50 = "aruco5x5_50"
    aruco5x5_100 = "aruco5x5_100"
    aruco5x5_250 = "aruco5x5_250"
    aruco5x5_1000 = "aruco5x5_1000"
    aruco6x6_50 = "aruco6x6_50"
    aruco6x6_100 = "aruco6x6_100"
    aruco6x6_250 = "aruco6x6_250"
    aruco6x6_1000 = "aruco6x6_1000"
    aruco7x7_50 = "aruco7x7_50"
    aruco7x7_100 = "aruco7x7_100"
    aruco7x7_250 = "aruco7x7_250"
    aruco7x7_1000 = "aruco7x7_1000"

    _valid_values = {
        "aruco4x4_50": _zivid.calibration.MarkerDictionary.aruco4x4_50,
        "aruco4x4_100": _zivid.calibration.MarkerDictionary.aruco4x4_100,
        "aruco4x4_250": _zivid.calibration.MarkerDictionary.aruco4x4_250,
        "aruco4x4_1000": _zivid.calibration.MarkerDictionary.aruco4x4_1000,
        "aruco5x5_50": _zivid.calibration.MarkerDictionary.aruco5x5_50,
        "aruco5x5_100": _zivid.calibration.MarkerDictionary.aruco5x5_100,
        "aruco5x5_250": _zivid.calibration.MarkerDictionary.aruco5x5_250,
        "aruco5x5_1000": _zivid.calibration.MarkerDictionary.aruco5x5_1000,
        "aruco6x6_50": _zivid.calibration.MarkerDictionary.aruco6x6_50,
        "aruco6x6_100": _zivid.calibration.MarkerDictionary.aruco6x6_100,
        "aruco6x6_250": _zivid.calibration.MarkerDictionary.aruco6x6_250,
        "aruco6x6_1000": _zivid.calibration.MarkerDictionary.aruco6x6_1000,
        "aruco7x7_50": _zivid.calibration.MarkerDictionary.aruco7x7_50,
        "aruco7x7_100": _zivid.calibration.MarkerDictionary.aruco7x7_100,
        "aruco7x7_250": _zivid.calibration.MarkerDictionary.aruco7x7_250,
        "aruco7x7_1000": _zivid.calibration.MarkerDictionary.aruco7x7_1000,
    }

    @classmethod
    def valid_values(cls):
        """Get valid values for MarkerDictionary.

        Returns:
            A list of strings representing valid values for MarkerDictionary.
        """
        return list(cls._valid_values.keys())

    @classmethod
    def marker_count(cls, dictionary_name):
        """Get the number of markers in a dictionary.

        Args:
            dictionary_name: Name of the dictionary, e.g. "aruco4x4_50". Must be one of the values returned by
                valid_values().

        Returns:
            Number of markers in the dictionary.

        Raises:
            ValueError: If the dictionary name is not one of the valid values returned by
                valid_values().
        """
        if dictionary_name not in cls._valid_values:
            raise ValueError(
                "Invalid dictionary name '{}'. Valid values are {}".format(
                    dictionary_name, cls.valid_values()
                )
            )

        return cls._valid_values[dictionary_name].marker_count()


class DetectionResultFiducialMarkers:
    """Class representing detected fiducial markers."""

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


def detect_calibration_board(source):
    """
    Detect feature points from a calibration board in a frame or using a given camera.

    If a camera is used, this function will perform a relatively slow but high-quality point cloud
    capture with the camera. This function is necessary for applications that require very
    high-accuracy DetectionResult, such as in-field verification/correction.

    The functionality is to be exclusively used in combination with Zivid verified calibration boards.
    For further information please visit https://support.zivid.com.

    Args:
        source: A frame containing an image of a calibration board or a camera pointed at
            a calibration board

    Raises:
        TypeError: If source is not of type Camera or Frame

    Returns:
        A DetectionResult instance
    """

    if isinstance(source, Camera):
        return DetectionResult(
            _zivid.calibration.detect_calibration_board(
                source._Camera__impl  # pylint: disable=protected-access
            )
        )
    if isinstance(source, Frame):
        return DetectionResult(
            _zivid.calibration.detect_calibration_board(
                source._Frame__impl  # pylint: disable=protected-access
            )
        )
    raise TypeError(
        "Unsupported type for argument source. Got {}, expected one of {}".format(
            type(source),
            (Camera, Frame),
        )
    )


def capture_calibration_board(camera):
    """
    Capture a calibration board with the given camera.

    The functionality is to be exclusively used in combination with Zivid verified calibration boards.
    For further information please visit https://support.zivid.com.

    This function will perform a relatively slow but high-quality point cloud capture with the
    given camera. This function is necessary for applications that require very high-accuracy
    captures, such as in-field verification/correction.

    The Frame that is returned from this function may be used as input to `detect_calibration_board`.
    You may also use `detect_calibration_board` directly, which will invoke this function under the
    hood and yield a DetectionResult.

    Args:
        camera: a Camera pointed at a calibration board

    Returns:
        A Frame
    """
    return Frame(
        _zivid.calibration.capture_calibration_board(
            camera._Camera__impl  # pylint: disable=protected-access
        )
    )


def detect_markers(frame, allowed_marker_ids, marker_dictionary):
    """Detect fiducial markers such as ArUco markers in a frame.

    Only markers with integer IDs are supported. To get more information about fiducial markers, refer to the
    wikipedia page:  https://en.wikipedia.org/wiki/Fiducial_marker

    For more information on ArUco markers specifically, see the OpenCV documentation on ArUco markers:
    https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html,

    Frame need not contain all markers listed in allowedMarkerIds for a successful detection.

    Args:
        frame: A frame containing an image of one or several fiducial markers
        allowed_marker_ids: List of the IDs of markers to be detected
        marker_dictionary: The name of the marker dictionary to use. The name must be one of the values returned by
            MarkerDictionary.valid_values()

    Raises:
        ValueError: If marker_dictionary is not one of the valid values returned by MarkerDictionary.valid_values()

    Returns:
        A DetectionResultFiducialMarkers instance
    """

    if marker_dictionary not in MarkerDictionary.valid_values():
        raise ValueError(
            "Invalid marker dictionary '{}'. Valid values are {}".format(
                marker_dictionary, MarkerDictionary.valid_values()
            )
        )
    dictionary = MarkerDictionary._valid_values.get(  # pylint: disable=protected-access
        marker_dictionary
    )

    return DetectionResultFiducialMarkers(
        _zivid.calibration.detect_markers(
            frame._Frame__impl,  # pylint: disable=protected-access
            allowed_marker_ids,
            dictionary,
        )
    )
