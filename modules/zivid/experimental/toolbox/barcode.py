"""Module for experimental barcode reading. This API may change in the future."""

import _zivid
from zivid.camera import Camera
from zivid.frame_2d import Frame2D
from zivid.settings2d import _to_settings2d


class LinearBarcodeFormat:
    """Collection of supported linear (1D) barcode formats."""

    code128 = "code128"
    code93 = "code93"
    code39 = "code39"
    ean13 = "ean13"
    ean8 = "ean8"
    upcA = "upcA"
    upcE = "upcE"

    _valid_values = {
        "code128": _zivid.toolbox.LinearBarcodeFormat.code128,
        "code93": _zivid.toolbox.LinearBarcodeFormat.code93,
        "code39": _zivid.toolbox.LinearBarcodeFormat.code39,
        "ean13": _zivid.toolbox.LinearBarcodeFormat.ean13,
        "ean8": _zivid.toolbox.LinearBarcodeFormat.ean8,
        "upcA": _zivid.toolbox.LinearBarcodeFormat.upcA,
        "upcE": _zivid.toolbox.LinearBarcodeFormat.upcE,
    }

    @classmethod
    def valid_values(cls):
        """List all valid linear barcode format values."""
        return cls._valid_values


class MatrixBarcodeFormat:
    """Collection of supported matrix (2D) barcode formats."""

    qrcode = "qrcode"
    dataMatrix = "dataMatrix"

    _valid_values = {
        "qrcode": _zivid.toolbox.MatrixBarcodeFormat.qrcode,
        "dataMatrix": _zivid.toolbox.MatrixBarcodeFormat.dataMatrix,
    }

    @classmethod
    def valid_values(cls):
        """List all valid matrix barcode format values."""
        return cls._valid_values


def _convert_to_internal_format_set(format_filter, format_class):
    format_set_internal = set()
    if format_filter is not None:
        valid_values = format_class.valid_values()
        for fmt in format_filter:
            if fmt not in valid_values:
                raise ValueError(
                    "Unsupported barcode format in format_filter: {}. Supported formats are: {}".format(
                        fmt, valid_values.keys()
                    )
                )
            format_set_internal.add(valid_values[fmt])
    return format_set_internal


class BarcodeDetectionResult:
    """Base class for barcode detection results implementation. Should not be used by the end-user."""

    def __init__(self, impl):
        """Initialize."""
        self.__impl = impl

    def code(self):
        """Code as string."""
        return self.__impl.code()

    def code_format(self):
        """Code format as string."""
        return self.__impl.code_format()

    def center_position(self):
        """Position of barcode as tuple of pixel coodinates (x,y)."""
        return tuple(self.__impl.center_position())

    def __str__(self):
        return str(self.__impl)


class LinearBarcodeDetectionResult(BarcodeDetectionResult):
    """Information about a detected linear (1D) barcode."""

    def __init__(self, impl):
        """Initialize LinearBarcodeDetectionResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.toolbox.LinearBarcodeDetectionResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.toolbox.LinearBarcodeDetectionResult
                )
            )
        super().__init__(impl)


class MatrixBarcodeDetectionResult(BarcodeDetectionResult):
    """Information about a detected matrix (2D) barcode."""

    def __init__(self, impl):
        """Initialize MatrixBarcodeDetectionResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.toolbox.MatrixBarcodeDetectionResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.toolbox.MatrixBarcodeDetectionResult
                )
            )
        super().__init__(impl)


class BarcodeDetector:
    """Class for enabling the detection of barcodes.

    Constructing an instance of this class initializes the resources needed for efficient barcode detection.
    For repeated detection it is recommended to keep and re-use an instance of this class and not create
    a new one every time.
    """

    def __init__(self):
        """Initialize BarcodeDetector."""
        self.__impl = _zivid.toolbox.BarcodeDetector()

    def suggest_settings(self, camera):
        """Get 2D capture settings that are ideal for barcode reading with the given camera.

        Args:
            camera: The camera to suggest settings for.

        Returns:
            A Settings2D instance with suggested settings for barcode reading.
        """
        if not isinstance(camera, Camera):
            raise TypeError("Unsupported type for argument camera. Got {}, expected {}".format(type(camera), Camera))
        settings2d_impl = self.__impl.suggest_settings(camera._Camera__impl)  # pylint: disable=protected-access
        return _to_settings2d(settings2d_impl)

    def read_linear_codes(self, frame2d, format_filter=None):
        """Detect and decode linear (1D) barcodes based on the result of a 2D capture.

        Args:
            frame2d: A Frame2D instance containing the image to find barcodes in.
            format_filter: An optional set of LinearBarcodeFormat values to filter the detection to only
                these formats. If None or an empty set is provided, all supported formats will be detected.

        Returns:
            A list of LinearBarcodeDetectionResult instances containing the results of the detection.
        """
        if not isinstance(frame2d, Frame2D):
            raise TypeError("Unsupported type for argument frame2d. Got {}, expected {}".format(type(frame2d), Frame2D))

        results = self.__impl.read_linear_codes(
            frame2d._Frame2D__impl,  # pylint: disable=protected-access
            _convert_to_internal_format_set(format_filter, LinearBarcodeFormat),
        )
        return [LinearBarcodeDetectionResult(result) for result in results]

    def read_matrix_codes(self, frame2d, format_filter=None):
        """Detect and decode matrix (2D) barcodes based on the result of a 2D capture.

        Args:
            frame2d: A Frame2D instance containing the image to find barcodes in.
            format_filter: An optional set of MatrixBarcodeFormat values to filter the detection to only
                these formats. If None or an empty set is provided, all supported formats will be detected.

        Returns:
            A list of MatrixBarcodeDetectionResult instances containing the results of the detection.
        """
        if not isinstance(frame2d, Frame2D):
            raise TypeError("Unsupported type for argument frame2d. Got {}, expected {}".format(type(frame2d), Frame2D))
        results = self.__impl.read_matrix_codes(
            frame2d._Frame2D__impl,  # pylint: disable=protected-access
            _convert_to_internal_format_set(format_filter, MatrixBarcodeFormat),
        )
        return [MatrixBarcodeDetectionResult(result) for result in results]

    def release(self):
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
