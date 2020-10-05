# pylint: disable=missing-module-docstring
import _zivid


class MultiCameraResidual:
    """Class representing the estimated errors of a multi-camera calibration."""

    def __init__(self, impl):  # noqa: D107
        if not isinstance(impl, _zivid.calibration.MultiCameraResidual):
            raise TypeError(
                "Unsupported type: {recieved_type}".format(recieved_type=type(impl))
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

    def __init__(self, impl):  # noqa: D107
        if not isinstance(impl, _zivid.calibration.MultiCameraOutput):
            raise TypeError(
                "Unsupported type: {recieved_type}".format(recieved_type=type(impl))
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
            List of MultiCameraResidual objects, one for each camera
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
        A MultiCameraOutput object
    """
    return MultiCameraOutput(
        _zivid.calibration.calibrate_multi_camera(
            [
                detection_result._DetectionResult__impl  # pylint: disable=protected-access
                for detection_result in detection_results
            ]
        )
    )
