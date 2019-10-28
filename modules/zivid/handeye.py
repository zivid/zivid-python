"""Contains hand-eye calibration functions and classes."""
import _zivid


def detect_feature_points(point_cloud):
    """Detect feature points from a calibration object in a point cloud.

    The functionality is to be exclusively used in combination with Zivid verified checkerboards.
    For further information please visit Zivid help page.

    Args:
        point_cloud: cloud from a frame containing an image of a calibration object

    Returns:
        Instance of DetectionResult

    """
    return DetectionResult(
        _zivid.handeye.detect_feature_points(
            point_cloud._PointCloud__impl  # pylint: disable=protected-access
        )
    )


def calibrate_eye_in_hand(calibration_inputs):
    """Perform eye-in-hand calibration.

    The procedure requires feature point sets acquired at the minimum from two poses.
    All the input poses have to be different. The feature point sets cannot be empty.
    All the feature point sets have to have same number of feature points.
    An exception will be thrown if the above requirements are not fulfilled.

    Args:
        calibration_inputs: a Sequence of CalibrationInput instances

    Returns:
        Instance of CalibrationOutput

    """
    return CalibrationOutput(
        _zivid.handeye.calibrate_eye_in_hand(
            [
                calib._CalibrationInput__impl  # pylint: disable=protected-access
                for calib in calibration_inputs
            ]
        )
    )


def calibrate_eye_to_hand(calibration_inputs):
    """Perform eye-to-hand calibration.

    The procedure requires feature point sets acquired at the minimum from two poses.
    All the input poses have to be different. The feature point sets cannot be empty.
    All the feature points have to have same number of elements.
    An exception will be thrown if the above requirements are not fulfilled.

    Args:
        calibration_inputs: a Sequence of CalibrationInput instances

    Returns:
        Instance of CalibrationOutput

    """
    return CalibrationOutput(
        _zivid.handeye.calibrate_eye_to_hand(
            [
                calib._CalibrationInput__impl  # pylint: disable=protected-access
                for calib in calibration_inputs
            ]
        )
    )


class Pose:  # pylint: disable=too-few-public-methods
    """Describes a robot pose."""

    def __init__(self, matrix):
        """Pose constructor taking a 4x4 transform.

        The constructor throws if the input transform does not describe pure rotation and translation.

        Args:
            matrix: a 4x4 numpy array

        """
        self.__impl = _zivid.handeye.Pose(matrix)

    def __str__(self):
        return self.__impl.to_string()


class DetectionResult:  # pylint: disable=too-few-public-methods
    """A result returned by the detect_feature_points(...) call."""

    def __init__(self, impl):
        """Initialize from internal DetectionResult.

        Args:
            impl: an internal DetectionResult Instance

        """
        self.__impl = impl

    def __str__(self):
        return self.__impl.to_string()

    def __bool__(self):
        return self.__impl.valid()


class CalibrationInput:  # pylint: disable=too-few-public-methods
    """Binds together a robot pose and the detection result acquired from the pose."""

    def __init__(self, pose, detected_features):
        """Binds together a robot pose and the detection result acquired from the pose.

        Args:
            pose: a robot pose
            detected_features: a DetectionResult instance

        """
        self.__impl = _zivid.handeye.CalibrationInput(
            pose._Pose__impl,  # pylint: disable=protected-access
            detected_features._DetectionResult__impl,  # pylint: disable=protected-access
        )

    def __str__(self):
        return self.__impl.to_string()


class CalibrationOutput:
    """The calibration result containing the computed pose and reprojection errors for all the input poses."""

    def __init__(self, internal_calibration_result):
        """Calibration results containing the computed pose and reprojection errors for all the input poses.

        Args:
            internal_calibration_result: An internal zivid calibration result instance

        """
        self.__impl = internal_calibration_result

    @property
    def per_pose_calibration_residuals(self):
        """Return the calibration residuals.

        Feature points (for each input pose) are transformed into a common frame.
        A rigid transform between feature points and corresponding centroids are utilized to
        compute residuals for rotational and translational parts. An exception is thrown if the result is not valid.

        Returns:
            a list of calibration residuals

        """
        return [
            CalibrationResidual(element)
            for element in self.__impl.perPoseCalibrationResiduals()
        ]

    @property
    def hand_eye_transform(self):
        """Hand-eye transform.

        A 4x4 matrix describing hand-eye calibration transform (either eye-in-hand or eye-to-hand).
        An exception is thrown if the result is not valid.

        Returns:
            A 4x4 numpy matrix

        """
        return self.__impl.handEyeTransform()

    @property
    def valid(self):
        """Test if CalibrationOutput is valid.

        Returns:
            a bool

        """
        return self.__impl.valid()

    def __bool__(self):
        return self.valid

    def __str__(self):
        return self.__impl.to_string()


class CalibrationResidual:
    """Binds together rotational and translational residual errors wrt. calibration transform."""

    def __init__(self, internal_calibration_residual):
        """Binds together rotational and translational residual errors wrt. calibration transform.

        Args:
            internal_calibration_residual: an internal calibration residual instance

        """
        self.__impl = internal_calibration_residual

    @property
    def rotation(self):
        """Component of calibration residual.

        Returns:
            Rotational residual in degrees.

        """
        return self.__impl.rotation()

    @property
    def translation(self):
        """Component of calibration residual.

        Returns:
             Translational residual in millimeters.

        """
        return self.__impl.translation()

    def __str__(self):
        return self.__impl.to_string()
