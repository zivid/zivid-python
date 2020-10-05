import _zivid
from zivid._calibration.pose import Pose
from zivid._calibration.detector import DetectionResult


class HandEyeInput:
    def __init__(self, robot_pose, detection_result):

        if not isinstance(robot_pose, Pose):
            raise TypeError(
                "Unsupported type for argument robot_pose. Expected zivid.calibration.Pose but got {}".format(
                    type(robot_pose)
                )
            )

        self.__impl = _zivid.calibration.HandEyeInput(
            robot_pose._Pose__impl,  # pylint: disable=protected-access
            detection_result._DetectionResult__impl,  # pylint: disable=protected-access
        )

    def robot_pose(self):
        return Pose(self.__impl.robot_pose())

    def detection_result(self):
        return DetectionResult(self.__impl.detection_result())

    def __str__(self):
        return str(self.__impl)


class HandEyeResidual:
    def __init__(self, internal_handeyeresidual):

        if not isinstance(internal_handeyeresidual, _zivid.calibration.HandEyeResidual):
            raise TypeError(
                "Unsupported type: {recieved_type}".format(
                    recieved_type=type(internal_handeyeresidual)
                )
            )

        self.__impl = internal_handeyeresidual

    def rotation(self):
        return self.__impl.rotation()

    def translation(self):
        return self.__impl.translation()

    def __str__(self):
        return str(self.__impl)


class HandEyeOutput:
    def __init__(self, internal_handeyeoutput):
        if not isinstance(internal_handeyeoutput, _zivid.calibration.HandEyeOutput):
            raise TypeError(
                "Unsupported type: {recieved_type}".format(
                    recieved_type=type(internal_handeyeoutput)
                )
            )

        self.__impl = internal_handeyeoutput

    def valid(self):
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def transform(self):
        return self.__impl.transform()

    def residuals(self):
        return [
            HandEyeResidual(internal_residual)
            for internal_residual in self.__impl.residuals()
        ]

    def __str__(self):
        return str(self.__impl)


def calibrate_eye_in_hand(calibration_inputs):
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_in_hand(
            [
                calibration_input._HandEyeInput__impl
                for calibration_input in calibration_inputs
            ]
        )
    )


def calibrate_eye_to_hand(calibration_inputs):
    return HandEyeOutput(
        _zivid.calibration.calibrate_eye_to_hand(
            [
                calibration_input._HandEyeInput__impl
                for calibration_input in calibration_inputs
            ]
        )
    )
