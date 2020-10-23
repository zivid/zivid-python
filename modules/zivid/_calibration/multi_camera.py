"""Module containing implementation of multi-camera calibration functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.calibration module.
"""

import _zivid


class MultiCameraResidual:
    """Class representing the estimated errors of a multi-camera calibration."""

    def __init__(self, impl):
        """Initialize MultiCameraResidual wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.MultiCameraResidual):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.calibration.MultiCameraResidual)
                )
            )
        self.__impl = impl

    def translation(self):
        """Get the average overlap error.

        Returns:
            Average overlap error in millimeters
        """
        return self.__impl.translation()

    def __str__(self):
        return str(self.__impl)


class MultiCameraOutput:
    """Class representing the result of a multi-camera calibration process."""

    def __init__(self, impl):
        """Initialize MultiCameraOutput wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.calibration.MultiCameraOutput):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.calibration.MultiCameraOutput)
                )
            )
        self.__impl = impl

    def valid(self):
        """Check validity of MultiCameraOutput.

        Returns:
            True if MultiCameraOutput is valid
        """
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def transforms(self):
        """Get multi-camera calibration transforms.

        Returns:
            List of 4x4 arrays, one for each camera
        """
        return self.__impl.transforms()

    def residuals(self):
        """Get multi-camera calibration residuals.

        Returns:
            List of MultiCameraResidual instances, one for each camera
        """
        return [
            MultiCameraResidual(internal_residual)
            for internal_residual in self.__impl.residuals()
        ]

    def __str__(self):
        return str(self.__impl)


def calibrate_multi_camera(detection_results):
    """Perform multi-camera calibration.

    Args:
        detection_results: List of DetectionResult, one for each camera

    Returns:
        A MultiCameraOutput instance
    """
    return MultiCameraOutput(
        _zivid.calibration.calibrate_multi_camera(
            [
                detection_result._DetectionResult__impl  # pylint: disable=protected-access
                for detection_result in detection_results
            ]
        )
    )
