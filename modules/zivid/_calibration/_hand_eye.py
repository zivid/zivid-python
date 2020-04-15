import _zivid
from zivid._calibration._pose import Pose
from zivid._calibration._detector import DetectionResult


class HandEyeInput:
    def __init__(self, robot_pose, detection_result):
        self.__impl = _zivid.calibration.HandEyeInput(
            robot_pose._Pose__impl,  # pylint: disable=protected-access
            detection_result._DetectionResult__impl,  # pylint: disable=protected-access
        )

    def pose(self):
        return Pose(self.__impl.pose)

    def detection_result(self):
        return DetectionResult(self.__impl.detection_result)

    def __str__(self):
        return str(self.__impl)


class HandEyeResidual:
    def __init__(self, rotation, translation):
        self.__impl = _zivid.calibration.HandEyeResidual(rotation, translation,)

    def rotation(self):
        return self.__impl.rotation

    def translation(self):
        return self.__impl.translation

    def __str__(self):
        return str(self.__impl)


class HandEyeOutput:
    def __init__(self, transform, residuals):
        self.__impl = _zivid.calibration.HandEyeOutput(transform, residuals)

    def valid(self):
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def transform(self):
        return self.__impl.transform

    def residuals(self):
        return self.__impl.residuals

    def __str__(self):
        return str(self.__impl)


def calibrate_eye_in_hand(calibration_inputs):
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_in_hand(
            [calibration_input for calibration_input in calibration_inputs]
        )
    )


def calibrate_eye_to_hand(calibration_inputs):
    return HandEyeOutput(_zivid.calibration.calibrate_eye_to_hand(calibration_inputs))
