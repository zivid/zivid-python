import pytest
import zivid
from zivid.experimental.toolbox.barcode import (
    BarcodeDetector,
    LinearBarcodeDetectionResult,
    LinearBarcodeFormat,
    MatrixBarcodeDetectionResult,
    MatrixBarcodeFormat,
)


def _check_barcode_detection_result(result):
    assert isinstance(str(result), str)
    assert isinstance(result.code(), str)
    assert isinstance(result.code_format(), str)
    assert isinstance(result.center_position(), tuple)
    assert len(result.center_position()) == 2
    assert isinstance(result.center_position()[0], float)
    assert isinstance(result.center_position()[1], float)


@pytest.mark.barcode_license
def test_read_linear_codes(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()
    results = detector.read_linear_codes(frame_2d)

    assert isinstance(results, list)
    assert len(results) == 15
    for result in results:
        assert isinstance(result, LinearBarcodeDetectionResult)
        _check_barcode_detection_result(result)


@pytest.mark.barcode_license
def test_read_matrix_codes(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()
    results = detector.read_matrix_codes(frame_2d)

    assert isinstance(results, list)
    assert len(results) == 15
    for result in results:
        assert isinstance(result, MatrixBarcodeDetectionResult)
        _check_barcode_detection_result(result)


@pytest.mark.barcode_license
def test_barcode_suggest_settings(file_camera):
    detector = BarcodeDetector()
    settings2d = detector.suggest_settings(file_camera)
    assert isinstance(settings2d, zivid.Settings2D)


@pytest.mark.barcode_license
def test_read_linear_codes_with_format_filter(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()

    # Empty format filter (should detect all formats)
    results = detector.read_linear_codes(frame_2d, format_filter=set())
    assert len(results) == 15

    # Single format matching codes in image
    results = detector.read_linear_codes(frame_2d, format_filter={LinearBarcodeFormat.code128})
    assert len(results) == 15

    # Single format not matching any codes in image
    results = detector.read_linear_codes(frame_2d, format_filter={LinearBarcodeFormat.ean13})
    assert len(results) == 0

    # Multiple formats, one matching codes in image
    results = detector.read_linear_codes(
        frame_2d,
        format_filter={
            LinearBarcodeFormat.code128,
            LinearBarcodeFormat.ean13,
            LinearBarcodeFormat.upcA,
        },
    )
    assert len(results) == 15

    # Multiple formats, none matching codes in image
    results = detector.read_linear_codes(
        frame_2d,
        format_filter={
            LinearBarcodeFormat.ean13,
            LinearBarcodeFormat.upcA,
        },
    )
    assert len(results) == 0

    # Specify formats as strings instead
    results = detector.read_linear_codes(
        frame_2d,
        format_filter={
            "code128",
            "ean13",
            "upcA",
        },
    )
    assert len(results) == 15

    # Error case: Specify string that is not a valid format
    with pytest.raises(ValueError, match="Unsupported barcode format in format_filter: notARealFormat."):
        detector.read_linear_codes(
            frame_2d,
            format_filter={
                "code128",
                "ean13",
                "notARealFormat",
                "upcA",
            },
        )


@pytest.mark.barcode_license
def test_read_matrix_codes_with_format_filter(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()

    # Empty format filter (should detect all formats)
    results = detector.read_matrix_codes(frame_2d, format_filter=set())
    assert len(results) == 15

    # Single format matching codes in image
    results = detector.read_matrix_codes(frame_2d, format_filter={MatrixBarcodeFormat.qrcode})
    assert len(results) == 15

    # Single format not matching any codes in image
    results = detector.read_matrix_codes(frame_2d, format_filter={MatrixBarcodeFormat.dataMatrix})
    assert len(results) == 0

    # Multiple formats, one matching codes in image
    results = detector.read_matrix_codes(
        frame_2d,
        format_filter={
            MatrixBarcodeFormat.dataMatrix,
            MatrixBarcodeFormat.qrcode,
        },
    )
    assert len(results) == 15

    # Specify formats as strings instead
    results = detector.read_matrix_codes(
        frame_2d,
        format_filter={
            "dataMatrix",
            "qrcode",
        },
    )
    assert len(results) == 15

    # Error case: Specify string that is not a valid format
    with pytest.raises(ValueError, match="Unsupported barcode format in format_filter: notARealFormat."):
        detector.read_matrix_codes(
            frame_2d,
            format_filter={
                "dataMatrix",
                "notARealFormat",
                "qrcode",
            },
        )
