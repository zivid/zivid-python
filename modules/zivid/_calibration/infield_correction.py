"""Module containing implementation of in-field correction functionality.

This module should not be imported directly by end-user, but rather accessed through
the zivid.experimental.calibration module.
"""

import _zivid
from zivid.calibration import DetectionResult


class InfieldCorrectionInput:
    """Container for input-data needed by in-field verification and correction functions."""

    def __init__(self, detection_result):
        """Construct an InfieldCorrectionInput instance.

        Input data should be captured by calling the version of detect_feature_points that
        takes a Camera argument.

        Args:
            detection_result: A DetectionResult instance
        Raises:
            TypeError: If one of the input arguments are of the wrong type
        """
        if not isinstance(detection_result, DetectionResult):
            raise TypeError(
                "Unsupported type for argument detection_result. Expected zivid.calibration.DetectionResult but got {}".format(
                    type(detection_result)
                )
            )
        self.__impl = _zivid.infield_correction.InfieldCorrectionInput(
            detection_result._DetectionResult__impl,  # pylint: disable=protected-access
        )

    def detection_result(self):
        """Get the contained DetectionResult

        Returns:
            A DetectionResult instance
        """
        return DetectionResult(self.__impl.detection_result())

    def valid(self):
        """Check if this object is valid for use with in-field correction.

        Returns:
            True if InfieldCorrectionInput is valid
        """
        return self.__impl.valid()

    def status_description(self):
        """Get a string describing the status of this input object.

        Mostly used to figure out why valid() is False.

        Returns:
            A human-readable string describing the status.
        """
        return self.__impl.status_description()

    def __bool__(self):
        return self.valid()

    def __str__(self):
        return str(self.__impl)


class CameraVerification:
    def __init__(self, impl):
        """Initialize CameraVerification wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.infield_correction.CameraVerification):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.infield_correction.CameraVerification)
                )
            )
        self.__impl = impl

    def local_dimension_trueness(self):
        return self.__impl.local_dimension_trueness()

    def position(self):
        return self.__impl.position()

    def __str__(self):
        return str(self.__impl)


class AccuracyEstimate:
    def __init__(self, impl):
        """Initialize AccuracyEstimate wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.infield_correction.AccuracyEstimate):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.infield_correction.AccuracyEstimate)
                )
            )
        self.__impl = impl

    def dimension_accuracy(self):
        return self.__impl.dimension_accuracy()

    def z_min(self):
        return self.__impl.z_min()

    def z_max(self):
        return self.__impl.z_max()

    def __str__(self):
        return str(self.__impl)


class CameraCorrection:
    def __init__(self, impl):
        """Initialize CameraCorrection wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.infield_correction.CameraCorrection):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.infield_correction.CameraCorrection)
                )
            )
        self.__impl = impl

    def accuracy_estimate(self):
        return AccuracyEstimate(self.__impl.accuracy_estimate())

    def __str__(self):
        return str(self.__impl)


def detect_feature_points(camera):

    return DetectionResult(
        _zivid.infield_correction.detect_feature_points_infield(
            camera._Camera__impl  # pylint: disable=protected-access
        )
    )


def verify_camera(infield_correction_input):

    return CameraVerification(
        _zivid.infield_correction.verify_camera(
            infield_correction_input._InfieldCorrectionInput__impl  # pylint: disable=protected-access
        )
    )


def compute_camera_correction(dataset):

    return CameraCorrection(
        _zivid.infield_correction.compute_camera_correction(
            [
                infield_correction_input._InfieldCorrectionInput__impl  # pylint: disable=protected-access
                for infield_correction_input in dataset
            ]
        )
    )


def write_camera_correction(camera, camera_correction):

    _zivid.infield_correction.write_camera_correction(
        camera._Camera__impl,  # pylint: disable=protected-access
        camera_correction._CameraCorrection__impl,  # pylint: disable=protected-access
    )


def reset_camera_correction(camera):

    _zivid.infield_correction.reset_camera_correction(
        camera._Camera__impl  # pylint: disable=protected-access
    )


def has_camera_correction(camera):

    return _zivid.infield_correction.has_camera_correction(
        camera._Camera__impl  # pylint: disable=protected-access
    )


def camera_correction_timestamp(camera):
    # Note: Local time

    return _zivid.infield_correction.camera_correction_timestamp(
        camera._Camera__impl  # pylint: disable=protected-access
    )
