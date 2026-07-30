"""Module for experimental barcode reading. This API may change in the future."""

from __future__ import annotations

import _zivid
from zivid.bounding_box import BoundingBox
from zivid.camera import Camera
from zivid.frame_2d import Frame2D
from zivid.settings2d import Settings2D, _to_settings2d


class LinearBarcodeFormat:
    """Collection of supported linear (1D) barcode formats."""

    code128 = "code128"
    code93 = "code93"
    code39 = "code39"
    ean13 = "ean13"
    ean8 = "ean8"
    upcA = "upcA"
    upcE = "upcE"
    itf = "itf"

    _valid_values = {
        "code128": _zivid.toolbox.LinearBarcodeFormat.code128,
        "code93": _zivid.toolbox.LinearBarcodeFormat.code93,
        "code39": _zivid.toolbox.LinearBarcodeFormat.code39,
        "ean13": _zivid.toolbox.LinearBarcodeFormat.ean13,
        "ean8": _zivid.toolbox.LinearBarcodeFormat.ean8,
        "upcA": _zivid.toolbox.LinearBarcodeFormat.upcA,
        "upcE": _zivid.toolbox.LinearBarcodeFormat.upcE,
        "itf": _zivid.toolbox.LinearBarcodeFormat.itf,
    }

    @classmethod
    def valid_values(cls) -> dict:
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
    def valid_values(cls) -> dict:
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

    def center_position(self) -> tuple:
        """Position of barcode as tuple of pixel coodinates (x,y)."""
        return tuple(self.__impl.center_position())

    def bounding_box(self) -> BoundingBox:
        """Get the bounding box of the region in the 2D image."""
        bb_impl = self.__impl.bounding_box()
        return BoundingBox(x=bb_impl.x, y=bb_impl.y, width=bb_impl.width, height=bb_impl.height)

    def __str__(self):
        return str(self.__impl)


class LinearBarcodeDetectionResult(BarcodeDetectionResult):
    """Information about an image region that is likely (but not guaranteed) to be a linear (1D) barcode."""

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


class BarcodeDecodingResult:
    """Base class for barcode decoding results implementation. Should not be used by the end-user."""

    def __init__(self, impl):
        """Initialize."""
        self.__impl = impl

    def code(self) -> str:
        """Code as string."""
        return self.__impl.code()

    def code_format(self) -> str:
        """Code format as string."""
        return self.__impl.code_format()

    def center_position(self) -> tuple:
        """Position of barcode as tuple of pixel coodinates (x,y)."""
        return tuple(self.__impl.center_position())

    def bounding_box(self) -> BoundingBox:
        """Get the bounding box of the barcode in the 2D image."""
        bb_impl = self.__impl.bounding_box()
        return BoundingBox(x=bb_impl.x, y=bb_impl.y, width=bb_impl.width, height=bb_impl.height)

    def __str__(self):
        return str(self.__impl)


class LinearBarcodeDecodingResult(BarcodeDecodingResult):
    """Information about a decoded linear (1D) barcode."""

    def __init__(self, impl):
        """Initialize LinearBarcodeDecodingResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.toolbox.LinearBarcodeDecodingResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.toolbox.LinearBarcodeDecodingResult
                )
            )
        super().__init__(impl)


class MatrixBarcodeDecodingResult(BarcodeDecodingResult):
    """Information about a decoded matrix (2D) barcode."""

    def __init__(self, impl):
        """Initialize MatrixBarcodeDecodingResult wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.toolbox.MatrixBarcodeDecodingResult):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), _zivid.toolbox.MatrixBarcodeDecodingResult
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

    def suggest_settings(self, camera: Camera) -> Settings2D:
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

    def detect_linear_codes(self, frame2d: Frame2D) -> list:
        """Detect linear (1D) barcode candidate regions based on the result of a 2D capture.

        This method detects potential barcode regions in the image but does not attempt to decode them.
        Since decoding has not yet been attempted, the list of regions returned are likely to contain some
        false positives (regions that look like barcodes but are not actually barcodes).
        Use decode_linear_codes to attempt to decode some or all candidate image regions.

        Args:
            frame2d: A Frame2D instance containing the image to find barcode candidates in.

        Returns:
            A list of LinearBarcodeDetectionResult instances (one per detected barcode candidate).
        """
        if not isinstance(frame2d, Frame2D):
            raise TypeError("Unsupported type for argument frame2d. Got {}, expected {}".format(type(frame2d), Frame2D))

        results = self.__impl.detect_linear_codes(
            frame2d._Frame2D__impl,  # pylint: disable=protected-access
        )
        return [LinearBarcodeDetectionResult(result) for result in results]

    def decode_linear_codes(self, detection_results: list, format_filter=None) -> list:
        """Decode linear (1D) barcode candidate regions.

        This method attempts to decode barcode candidates that were previously detected using
        detect_linear_codes. The ordering of the returned decoding results matches the ordering
        of the input detection results. If a specific candidate failed to decode, the corresponding
        element in the returned list will be None.

        Args:
            detection_results: A list of LinearBarcodeDetectionResult instances from a previous call
                to detect_linear_codes.
            format_filter: An optional set of LinearBarcodeFormat values to filter the detection to only
                these formats. If None or an empty set is provided, all supported formats will be detected.

        Returns:
            A list of optional LinearBarcodeDecodingResult instances (None if decoding failed for that candidate).
        """
        if not isinstance(detection_results, list):
            raise TypeError(
                "Unsupported type for argument detection_results. Got {}, expected list".format(type(detection_results))
            )

        # Extract the internal implementations from the detection results
        detection_results_impl = [
            result._BarcodeDetectionResult__impl for result in detection_results  # pylint: disable=protected-access
        ]

        results = self.__impl.decode_linear_codes(
            detection_results_impl,
            _convert_to_internal_format_set(format_filter, LinearBarcodeFormat),
        )
        return [LinearBarcodeDecodingResult(result) if result is not None else None for result in results]

    def read_linear_codes(self, frame2d: Frame2D, format_filter=None) -> list:
        """Detect and decode linear (1D) barcodes based on the result of a 2D capture.

        Args:
            frame2d: A Frame2D instance containing the image to find barcodes in.
            format_filter: An optional set of LinearBarcodeFormat values to filter the detection to only
                these formats. If None or an empty set is provided, all supported formats will be detected.

        Returns:
            A list of LinearBarcodeDecodingResult instances containing the results of the detection.
        """
        if not isinstance(frame2d, Frame2D):
            raise TypeError("Unsupported type for argument frame2d. Got {}, expected {}".format(type(frame2d), Frame2D))

        results = self.__impl.read_linear_codes(
            frame2d._Frame2D__impl,  # pylint: disable=protected-access
            _convert_to_internal_format_set(format_filter, LinearBarcodeFormat),
        )
        return [LinearBarcodeDecodingResult(result) for result in results]

    def read_matrix_codes(self, frame2d: Frame2D, format_filter=None) -> list:
        """Detect and decode matrix (2D) barcodes based on the result of a 2D capture.

        Args:
            frame2d: A Frame2D instance containing the image to find barcodes in.
            format_filter: An optional set of MatrixBarcodeFormat values to filter the detection to only
                these formats. If None or an empty set is provided, all supported formats will be detected.

        Returns:
            A list of MatrixBarcodeDecodingResult instances containing the results of the decoding.
        """
        if not isinstance(frame2d, Frame2D):
            raise TypeError("Unsupported type for argument frame2d. Got {}, expected {}".format(type(frame2d), Frame2D))
        results = self.__impl.read_matrix_codes(
            frame2d._Frame2D__impl,  # pylint: disable=protected-access
            _convert_to_internal_format_set(format_filter, MatrixBarcodeFormat),
        )
        return [MatrixBarcodeDecodingResult(result) for result in results]

    def release(self) -> None:
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
