"""Experimental implementation of hand-eye calibration for robots with low degrees-of-freedom.

This API may change in the future.
"""

import collections.abc
import _zivid
from zivid.calibration import Pose, HandEyeOutput, MarkerDictionary


class FixedPlacementOfFiducialMarker:
    """Specifies the fixed placement of a fiducial marker for low degrees-of-freedom hand-eye calibration."""

    def __init__(self, marker_id, position):
        """Construct a FixedPlacementOfFiducialMarker.

        For eye-in-hand calibration, positions should be given in the robot's base frame. For eye-to-hand calibration,
        positions should be given in the robot's end-effector frame.

        Note: the units of the input robot poses must be consistent with the units of the point clouds used to create
        the detection result. Zivid point clouds are, by default, in millimeters.

        Args:
            marker_id: The ID of the fiducial marker to specify a position for.
            position: The position of the fiducial marker as a three-element list, specified at the center of the
                marker.

        Raises:
            TypeError: If one of the input arguments is of the wrong type.
        """
        if not isinstance(marker_id, int):
            raise TypeError(
                "Unsupported type for argument marker_id. Expected int but got {}".format(
                    type(marker_id)
                )
            )

        if not isinstance(
            position, (collections.abc.Iterable, _zivid.data_model.PointXYZ)
        ):
            raise TypeError(
                "Unsupported type for argument position. Expected: (collections.abc.Iterable, _zivid.data_model.PointXYZ), got {value_type}".format(
                    value_type=type(position)
                )
            )

        self.__impl = _zivid.calibration.FixedPlacementOfFiducialMarker(
            marker_id,  # pylint: disable=protected-access
            _zivid.data_model.PointXYZ(position),  # pylint: disable=protected-access
        )

    @property
    def id(self):
        """Get ID of fiducial marker.

        Returns:
            An integer representing the ID
        """
        return self.__impl.id

    @property
    def position(self):
        """Get position of fiducial marker.

        Returns:
            A three-element list of floats
        """
        return self.__impl.position.to_array()

    def __str__(self):
        return str(self.__impl)


class FixedPlacementOfFiducialMarkers:  # pylint: disable=too-few-public-methods
    """Specifies the fixed placement of a list of fiducial markers for low degrees-of-freedom hand-eye calibration."""

    def __init__(self, marker_dictionary, markers):
        """Construct a FixedPlacementOfFiducialMarkers instance.

        Args:
            marker_dictionary: The dictionary that describes the appearance of the given markers. The name must be one
                of the values returned by MarkerDictionary.valid_values()
            markers: A list of FixedPlacementOfFiducialMarker describing the fixed placement of fiducial markers.

        Raises:
            ValueError: If marker_dictionary is not one of the valid values returned by MarkerDictionary.valid_values()
            TypeError: If one of the input arguments are of the wrong type
        """

        if marker_dictionary not in MarkerDictionary.valid_values():
            raise ValueError(
                "Invalid marker dictionary '{}'. Valid values are {}".format(
                    marker_dictionary, MarkerDictionary.valid_values()
                )
            )

        dictionary = (
            MarkerDictionary._valid_values.get(  # pylint: disable=protected-access
                marker_dictionary
            )
        )

        if not (
            isinstance(markers, list)
            and all(
                isinstance(marker, FixedPlacementOfFiducialMarker) for marker in markers
            )
        ):
            raise TypeError(
                "Unsupported type for argument position. Expected list of FixedPlacementOfFiducialMarker but got {}".format(
                    type(markers)
                )
            )

        self.__impl = _zivid.calibration.FixedPlacementOfFiducialMarkers(
            dictionary,  # pylint: disable=protected-access
            [
                marker._FixedPlacementOfFiducialMarker__impl  # pylint: disable=protected-access
                for marker in markers
            ],
        )

    def __str__(self):
        return str(self.__impl)


class FixedPlacementOfCalibrationBoard:  # pylint: disable=too-few-public-methods
    """Specifies the fixed placement of a Zivid calibration board for low degrees-of-freedom hand-eye calibration."""

    def __init__(self, position_or_pose):
        """Construct a FixedPlacementOfCalibrationBoard instance.

        For eye-in-hand calibration, the position or pose should be given in the robot's base frame. For eye-to-hand
        calibration, the position or pose should be given in the robot's end-effector frame.

        The origin is the top left inner corner of the calibration board. Using a pose instead of a position can improve
        accuracy of the hand-eye calibration in some situations.

        Note: the units of the input robot poses must be consistent with the units of the point clouds used to create
        the detection result. Zivid point clouds are, by default, in millimeters.

        Args:
            position_or_pose: A position specifying the origin of the calibration board as a three-element list, or the
                pose of the calibration board specified using the Pose type.

        Raises:
            TypeError: If the input argument is of the wrong type.
        """
        if isinstance(position_or_pose, Pose):
            self.__impl = _zivid.calibration.FixedPlacementOfCalibrationBoard(
                position_or_pose._Pose__impl  # pylint: disable=protected-access
            )
        elif isinstance(
            position_or_pose, (collections.abc.Iterable, _zivid.data_model.PointXYZ)
        ):
            self.__impl = _zivid.calibration.FixedPlacementOfCalibrationBoard(
                _zivid.data_model.PointXYZ(
                    position_or_pose
                ),  # pylint: disable=protected-access
            )
        else:
            raise TypeError(
                "Unsupported type for argument id. Expected zivid.calibration.Pose "
                "or a three-element list, but got {}".format(type(position_or_pose))
            )

    def __str__(self):
        return str(self.__impl)


class FixedPlacementOfCalibrationObjects:  # pylint: disable=too-few-public-methods
    """Specifies the fixed placement of calibration objects for low degrees-of-freedom hand-eye calibration."""

    def __init__(self, fixed_objects):
        """Construct a FixedPlacementOfCalibrationObjects instance from fiducial markers or a calibration board.

        Args:
            fixed_objects: An instance of FixedPlacementOfFiducialMarkers or FixedPlacementOfCalibrationBoard.

        Raises:
            TypeError: If the input argument is of the wrong type.
        """
        if isinstance(fixed_objects, FixedPlacementOfFiducialMarkers):
            self.__impl = _zivid.calibration.FixedPlacementOfCalibrationObjects(
                fixed_objects._FixedPlacementOfFiducialMarkers__impl  # pylint: disable=protected-access
            )
        elif isinstance(fixed_objects, FixedPlacementOfCalibrationBoard):
            self.__impl = _zivid.calibration.FixedPlacementOfCalibrationObjects(
                fixed_objects._FixedPlacementOfCalibrationBoard__impl  # pylint: disable=protected-access
            )
        else:
            raise TypeError(
                "Unsupported type for argument fixed_objects. Got {}, expected {} or {}".format(
                    type(fixed_objects),
                    FixedPlacementOfFiducialMarkers,
                    FixedPlacementOfCalibrationBoard,
                )
            )

    def __str__(self):
        return str(self.__impl)


def calibrate_eye_in_hand_low_dof(calibration_inputs, fixed_objects):
    """Perform eye-in-hand calibration for low degrees-of-freedom robots.

    For robots with low degrees-of-freedom (DOF), that is, less than 6 DOF, the robot pose and capture inputs are not
    alone sufficient to uniquely identify the solution to the hand-eye calibration. This procedure additionally takes
    knowledge about the fixed placement of the calibration objects in the scene to provide a unique solution. For 6 DOF
    robots, consider using the `calibrate_eye_in_hand` function instead.

    The procedure requires all robot poses to be different. At least 2 poses are required when using a calibration
    board, or 6 poses when using fiducial markers. For fiducial markers, each marker must be detected across 2 poses at
    minimum. An exception will be thrown if the preceding requirements are not fulfilled.

    Note: the units of the input robot poses must be consistent with the units of the point clouds used to create the
    detection results. Zivid point clouds are, by default, in millimeters.

    Args:
        calibration_inputs: List of HandEyeInput
        fixed_objects: Specifies the fixed placement of calibration objects in the robot's base frame, using an instance
            of FixedPlacementOfCalibrationObjects.

    Returns:
        A HandEyeOutput instance containing the eye-in-hand transform (camera pose in robot end-effector frame)
    """
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_in_hand_low_dof(
            [
                calibration_input._HandEyeInput__impl  # pylint: disable=protected-access
                for calibration_input in calibration_inputs
            ],
            fixed_objects._FixedPlacementOfCalibrationObjects__impl,  # pylint: disable=protected-access
        )
    )


def calibrate_eye_to_hand_low_dof(calibration_inputs, fixed_objects):
    """Perform eye-to-hand calibration for low degrees-of-freedom robots.

    For robots with low degrees-of-freedom (DOF), that is, less than 6 DOF, the robot pose and capture inputs are not
    alone sufficient to uniquely identify the solution to the hand-eye calibration. This procedure additionally takes
    knowledge about the fixed placement of the calibration objects in the scene to provide a unique solution. For 6 DOF
    robots, consider using the `calibrate_eye_to_hand` function instead.

    The procedure requires all robot poses to be different. At least 2 poses are required when using a calibration
    board, or 6 poses when using fiducial markers. For fiducial markers, each marker must be detected across 2 poses at
    minimum. An exception will be thrown if the preceding requirements are not fulfilled.

    Note: the units of the input robot poses must be consistent with the units of the point clouds used to create the
    detection results. Zivid point clouds are, by default, in millimeters.

    Args:
        calibration_inputs: List of HandEyeInput
        fixed_objects: Specifies the fixed placement of calibration objects in the robot's end-effector frame, using an
            instance of FixedPlacementOfCalibrationObjects.

    Returns:
        A HandEyeOutput instance containing the eye-to-hand transform (camera pose in robot base frame)
    """
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_to_hand_low_dof(
            [
                calibration_input._HandEyeInput__impl  # pylint: disable=protected-access
                for calibration_input in calibration_inputs
            ],
            fixed_objects._FixedPlacementOfCalibrationObjects__impl,  # pylint: disable=protected-access
        )
    )
