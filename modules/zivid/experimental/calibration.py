"""Module for experimental calibration features. This API may change in the future."""

import _zivid
from zivid.experimental import PixelMapping
from zivid.calibration import DetectionResult
from zivid.camera import Camera
from zivid.camera_intrinsics import _to_camera_intrinsics
from zivid.frame import Frame
from zivid.settings import Settings, _to_internal_settings
from zivid.settings2d import Settings2D, _to_internal_settings2d


def intrinsics(camera, settings=None):
    """Get intrinsic parameters of a given camera and settings (3D or 2D).

    These intrinsic parameters take into account the expected resolution of the point clouds captured
    with the given settings. If settings are not provided, intrinsics appropriate for the camera's
    default 3D capture settings is returned.

    For a 2D+3D capture, the 2D color image and 3D point cloud may have different resolutions, depending
    on the pixel sampling and resampling settings used in the Settings and in the Settings.color. This function
    returns intrinsics applicable for the 3D point cloud resolution. You can call this function with a Settings2D
    instance for the 2D intrinsics.

    Args:
        camera: A Camera instance
        settings: Settings or Settings2D to be used to get correct intrinsics (optional)

    Returns:
        A CameraIntrinsics instance

    Raises:
        TypeError: If settings argument is not Settings or Settings2D
    """
    if settings is None:
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(
                camera._Camera__impl  # pylint: disable=protected-access
            )
        )
    if isinstance(settings, Settings):
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(
                camera._Camera__impl,  # pylint: disable=protected-access
                _to_internal_settings(settings),
            )
        )
    if isinstance(settings, Settings2D):
        return _to_camera_intrinsics(
            _zivid.calibration.intrinsics(
                camera._Camera__impl,  # pylint: disable=protected-access
                _to_internal_settings2d(settings),
            )
        )
    raise TypeError(
        "Unsupported type for argument settings. Got {}, expected Settings or Settings2D.".format(
            type(settings)
        )
    )


def estimate_intrinsics(frame):
    """Estimate camera intrinsics for a given frame.

    The estimated parameters may be used to project 3D point cloud onto the corresponding 2D image.
    This function is for advanced use cases. Otherwise, use intrinsics(camera).

    For a 2D+3D capture, the 2D color image and 3D point cloud may have different resolutions, depending
    on the pixel sampling and resampling settings used in the Settings and in the Settings.color. This function
    returns intrinsics applicable for the 3D point cloud resolution.

    Args:
        frame: A Frame instance

    Returns:
        A CameraIntrinsics instance
    """
    return _to_camera_intrinsics(
        _zivid.calibration.estimate_intrinsics(
            frame._Frame__impl  # pylint: disable=protected-access
        )
    )


def pixel_mapping(camera, settings):
    """Return pixel mapping information given camera and settings.

    When mapping from a subsampled point cloud to a full resolution
    2D image it is important to get the pixel mapping correct. This
    mapping depends on camera model and settings. This function provides
    the correct parameters to map the 2D coordinates in a point cloud
    captured using `settings` to the full resolution of the camera.

    Args:
        camera: Reference to camera instance.
        settings: Reference to settings instance.

    Returns:
        A PixelMapping instance.
    """

    pixel_mapping_handle = _zivid.calibration.pixel_mapping(
        camera._Camera__impl,  # pylint: disable=protected-access
        _to_internal_settings(settings),
    )
    return PixelMapping(
        pixel_mapping_handle.row_stride(),
        pixel_mapping_handle.col_stride(),
        pixel_mapping_handle.row_offset(),
        pixel_mapping_handle.col_offset(),
    )


def capture_calibration_board(camera):
    """Capture the calibration board.

    This function is used to capture an official Zivid calibration board, using settings that are
    optimized for the board. This method must be used to capture the Frame that is given as input to
    detect_feature_points(frame).

    You can also call detect_feature_points(camera), providing a camera as the first argument, and
    it will call this function for you.

    To verify if you successfully captured the calibration board and found the feature points, call
    detection_result = detect_feature_points(frame) with the frame returned from this function, then
    check detection_result.valid() on the returned DetectionResult object.

    Args:
        camera: A camera that is pointing at a calibration checkerboard.

    Returns:
        A Frame that can be used with detect_feature_points.
    """
    return Frame(
        _zivid.infield_correction.capture_calibration_board(
            camera._Camera__impl  # pylint: disable=protected-access
        )
    )


def detect_feature_points(source):
    """Detect feature points from a calibration object.

    Using this version of the detectFeaturePoints function is necessary to
    ensure that the data quality is sufficient for use in in-field verification
    and correction.

    The functionality is to be exclusively used in combination with Zivid
    verified checkerboards.

    Args:
        source: A Camera that is pointing at a calibration checkerboard
                or a Frame that was captured using capture_calibration_board.

    Returns:
        A DetectionResult instance.

    Raises:
        TypeError: If source is not a Camera or a Frame.
    """
    if isinstance(source, Camera):
        _source = source._Camera__impl  # pylint: disable=protected-access
    elif isinstance(source, Frame):
        _source = source._Frame__impl  # pylint: disable=protected-access
    else:
        raise TypeError(
            "Unsupported type for argument source. Got {}, expected {}, {}".format(
                type(source), Camera, Frame
            )
        )
    return DetectionResult(
        _zivid.infield_correction.detect_feature_points_infield(_source)
    )


def verify_camera(infield_correction_input):
    """Verify the current camera trueness based on a single measurement.

    The purpose of this function is to allow quick assessment of the quality of
    the in-field correction on a camera (or the need for one if none exists
    already). This function will throw an exception if any of the provided
    InfieldCorrectionInput have valid()==False.

    The return value of this function will give an indication of the dimension
    trueness at the location where the input data was captured. If the returned
    assessment indicates a trueness error that is above threshold for your
    application, consider using compute_camera_correction in order to get an
    updated correction for the camera.

    Args:
        infield_correction_input: An InfieldCorrectionInput instance.

    Returns:
        A CameraVerification instance.
    """
    return CameraVerification(
        _zivid.infield_correction.verify_camera(
            infield_correction_input._InfieldCorrectionInput__impl  # pylint: disable=protected-access
        )
    )


def compute_camera_correction(dataset):
    """Calculate new in-field camera correction.

    The purpose of this function is to calculate a new in-field correction for a
    camera based on a series of calibration object captures taken at varying
    distances. This function will throw an exception if any of the provided
    InfieldCorrectionInput have valid()==False.

    The quantity and range of data is up to the user, but generally a larger
    dataset will yield a more accurate and reliable correction. If all
    measurements were taken at approximately the same distance, the resulting
    correction will mainly be valid at those distances. If several measurements
    were taken at significantly different distances, the resulting correction
    will likely be more suitable for extrapolation to distances beyond where the
    dataset was collected.

    The result of this process is a CameraCorrection object, which will contain
    information regarding the proposed working range and the accuracy that can
    be expected within the working range, if the correction is written to the
    camera. The correction may be written to the camera using
    writeCameraCorrection.

    This function will throw an exception if the input data is extremely
    inconsistent/noisy.

    Args:
        dataset: A list of InfieldCorrectionInput instances.

    Returns:
        A CameraCorrection instance.
    """
    return CameraCorrection(
        _zivid.infield_correction.compute_camera_correction(
            [
                infield_correction_input._InfieldCorrectionInput__impl  # pylint: disable=protected-access
                for infield_correction_input in dataset
            ]
        )
    )


def write_camera_correction(camera, camera_correction):
    """Write the in-field correction on a camera.

    After calling this function, the given correction will automatically be used
    any time the capture function is called on this camera. The correction will
    be persisted on the camera even though the camera is power-cycled or
    connected to a different PC.

    Beware that calling this will overwrite any existing correction present on
    the camera.

    Args:
        camera: The Camera to write the correction to.
        camera_correction: The CameraCorrection instance to write to the camera.
    """
    _zivid.infield_correction.write_camera_correction(
        camera._Camera__impl,  # pylint: disable=protected-access
        camera_correction._CameraCorrection__impl,  # pylint: disable=protected-access
    )


def reset_camera_correction(camera):
    """Reset the in-field correction on a camera to factory settings.

    Args:
        camera: The Camera to reset.
    """
    _zivid.infield_correction.reset_camera_correction(
        camera._Camera__impl  # pylint: disable=protected-access
    )


def has_camera_correction(camera):
    """Check if the camera has an in-field correction written to it.

    This is false if write_camera_correction has never been called using this
    camera. It will also be false after calling reset_camera_correction.

    Args:
        camera: The Camera to check.

    Returns:
        Boolean indicating whether or not the camera has an in-field correction.
    """
    return _zivid.infield_correction.has_camera_correction(
        camera._Camera__impl  # pylint: disable=protected-access
    )


def camera_correction_timestamp(camera):
    """Get the time at which the camera's in-field correction was created.

    Args:
        camera: The Camera to check.

    Returns:
        A timestamp indicating when the correction was created.
    """

    return _zivid.infield_correction.camera_correction_timestamp(
        camera._Camera__impl  # pylint: disable=protected-access
    )


class InfieldCorrectionInput:
    """Container for input-data needed by in-field verification and correction functions."""

    def __init__(self, detection_result):
        """Construct an InfieldCorrectionInput instance.

        Input data should be captured by calling the version of detect_feature_points that
        takes a Camera or Frame argument.

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
        """Get the contained DetectionResult.

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
    """An assessment of the current dimension trueness of a camera at a specific location."""

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
                    type(impl), _zivid.infield_correction.CameraVerification
                )
            )
        self.__impl = impl

    def local_dimension_trueness(self):
        """Get the estimated local dimension trueness.

        The dimension trueness represents the relative deviation between the
        measured size of the calibration object and the true size of the
        calibration object, including the effects of any in-field correction
        stored on the camera at the time of capture. Note that this estimate is
        local, i.e. only valid for the region of space very close to the
        calibration object.

        The returned value is a fraction (relative trueness error). Multiply by
        100 to get trueness in percent.

        Returns:
            Estimated local dimension trueness.
        """
        return self.__impl.local_dimension_trueness()

    def position(self):
        """Get the location at which the measurement was made.

        Returns:
            Location (XYZ) in the camera reference frame.
        """
        return self.__impl.position()

    def __str__(self):
        return str(self.__impl)


class AccuracyEstimate:
    """A dimension accuracy estimate for a specific working volume."""

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
                    type(impl), _zivid.infield_correction.AccuracyEstimate
                )
            )
        self.__impl = impl

    def dimension_accuracy(self):
        """Get the estimated dimension accuracy obtained if the correction is applied.

        This number represents a 1-sigma (68% confidence) upper bound for
        dimension trueness error in the working volume (z=zMin() to z=zMax(),
        across the entire field of view). In other words, it represents the
        expected distribution of local dimension trueness measurements (see
        CameraVerification) that can be expected if measuring throughout the
        working volume.

        The returned value is a fraction (relative trueness error). Multiply by
        100 to get trueness in percent.

        Note that the accuracy close to where the original data was captured is
        likely much better than what is implied by this number. This number is
        rather an accuracy estimate for the entire extrapolated working volume.

        Returns:
            A 1-sigma (68% confidence) upper bound for trueness error in the working volume.
        """
        return self.__impl.dimension_accuracy()

    def z_min(self):
        """Get the range of validity of the accuracy estimate (lower end).

        Returns:
            Minimum z-value of working volume in millimeters.
        """
        return self.__impl.z_min()

    def z_max(self):
        """Get the range of validity of the accuracy estimate (upper end).

        Returns:
            Maximum z-value of working volume in millimeters.
        """
        return self.__impl.z_max()

    def __str__(self):
        return str(self.__impl)


class CameraCorrection:
    """An in-field correction that may be written to a camera."""

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
                    type(impl), _zivid.infield_correction.CameraCorrection
                )
            )
        self.__impl = impl

    def accuracy_estimate(self):
        """Get an estimate for expected dimension accuracy if the correction is applied to the camera.

        Returns:
            An AccuracyEstimate instance.
        """
        return AccuracyEstimate(self.__impl.accuracy_estimate())

    def __str__(self):
        return str(self.__impl)
