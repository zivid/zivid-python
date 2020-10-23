"""Module containing implementation of hand-eye calibration functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""

import _zivid
from zivid._calibration.pose import Pose
from zivid._calibration.detector import DetectionResult


class HandEyeInput:
    """Class binding together a robot pose and the corresponding detection result."""

    def __init__(self, robot_pose, detection_result):
        """Construct a HandEyeInput.

        Args:
            robot_pose: The robot Pose at the time of capture
            detection_result: The DetectionResult captured when in the above pose

        Raises:
            TypeError: If one of the input arguments are of the wrong type
        """
        if not isinstance(robot_pose, Pose):
            raise TypeError(
                "Unsupported type for argument robot_pose. Expected zivid.calibration.Pose but got {}".format(
                    type(robot_pose)
                )
            )
        if not isinstance(detection_result, DetectionResult):
            raise TypeError(
                "Unsupported type for argument detection_result. Expected zivid.calibration.DetectionResult but got {}".format(
                    type(detection_result)
                )
            )
        self.__impl = _zivid.calibration.HandEyeInput(
            robot_pose._Pose__impl,  # pylint: disable=protected-access
            detection_result._DetectionResult__impl,  # pylint: disable=protected-access
        )

    def robot_pose(self):
        """Get the contained robot pose.

        Returns:
            A Pose instance
        """
        return Pose(self.__impl.robot_pose())

    def detection_result(self):
        """Get the contained detection result.

        Returns:
            A DetectionResult instance
        """
        return DetectionResult(self.__impl.detection_result())

    def __str__(self):
        return str(self.__impl)


class HandEyeResidual:
    """Class representing the estimated errors of a calibrated hand-eye transform."""

    def __init__(self, impl):
        """Initialize HandEyeResidual wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.HandEyeResidual):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.calibration.HandEyeResidual)
                )
            )
        self.__impl = impl

    def rotation(self):
        """Get the rotation residual.

        Returns:
            Rotation residual in degrees
        """
        return self.__impl.rotation()

    def translation(self):
        """Get the translation residual.

        Returns:
            Translation residual in millimeters
        """
        return self.__impl.translation()

    def __str__(self):
        return str(self.__impl)


class HandEyeOutput:
    """Class representing the result of a hand-eye calibration process."""

    def __init__(self, impl):
        """Initialize HandEyeOutput wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.HandEyeOutput):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.calibration.HandEyeOutput)
                )
            )
        self.__impl = impl

    def valid(self):
        """Check validity of HandEyeOutput.

        Returns:
            True if HandEyeOutput is valid
        """
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def transform(self):
        """Get hand-eye transform.

        Returns:
            A 4x4 array representing a hand-eye transform
        """
        return self.__impl.transform()

    def residuals(self):
        """Get hand-eye calibration residuals.

        Returns:
            List of HandEyeResidual, one for each pose.
        """
        return [
            HandEyeResidual(internal_residual)
            for internal_residual in self.__impl.residuals()
        ]

    def __str__(self):
        return str(self.__impl)


def calibrate_eye_in_hand(calibration_inputs):
    """Perform eye-in-hand calibration.

    Args:
        calibration_inputs: List of HandEyeInput

    Returns:
        A HandEyeOutput instance containing the eye-in-hand transform
    """
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_in_hand(
            [
                calibration_input._HandEyeInput__impl  # pylint: disable=protected-access
                for calibration_input in calibration_inputs
            ]
        )
    )


def calibrate_eye_to_hand(calibration_inputs):
    """Perform eye-to-hand calibration.

    Args:
        calibration_inputs: List of HandEyeInput

    Returns:
        A HandEyeOutput instance containing the eye-to-hand transform
    """
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_to_hand(
            [
                calibration_input._HandEyeInput__impl  # pylint: disable=protected-access
                for calibration_input in calibration_inputs
            ]
        )
    )
