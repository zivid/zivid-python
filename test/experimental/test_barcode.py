import pytest
import zivid
from zivid.experimental.toolbox.barcode import (
    BarcodeDetector,
    LinearBarcodeDecodingResult,
    LinearBarcodeDetectionResult,
    LinearBarcodeFormat,
    MatrixBarcodeDecodingResult,
    MatrixBarcodeFormat,
)


def _check_bounding_box(bb, center_position):
    assert isinstance(str(bb), str)
    assert isinstance(bb, zivid.BoundingBox)
    assert isinstance(bb.x, int)
    assert isinstance(bb.y, int)
    assert isinstance(bb.width, int)
    assert isinstance(bb.height, int)
    assert center_position[0] >= bb.x
    assert center_position[0] <= bb.x + bb.width
    assert center_position[1] >= bb.y
    assert center_position[1] <= bb.y + bb.height


def _check_barcode_detection_result(result):
    assert isinstance(str(result), str)
    assert isinstance(result.center_position(), tuple)
    assert len(result.center_position()) == 2
    assert isinstance(result.center_position()[0], float)
    assert isinstance(result.center_position()[1], float)
    _check_bounding_box(result.bounding_box(), result.center_position())


def _check_barcode_decoding_result(result):
    assert isinstance(str(result), str)
    assert isinstance(result.code(), str)
    assert isinstance(result.code_format(), str)
    assert isinstance(result.center_position(), tuple)
    assert len(result.center_position()) == 2
    assert isinstance(result.center_position()[0], float)
    assert isinstance(result.center_position()[1], float)
    _check_bounding_box(result.bounding_box(), result.center_position())


@pytest.mark.barcode_license
def test_read_linear_codes(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()
    results = detector.read_linear_codes(frame_2d)

    assert isinstance(results, list)
    assert len(results) == 15
    for result in results:
        assert isinstance(result, LinearBarcodeDecodingResult)
        _check_barcode_decoding_result(result)


@pytest.mark.barcode_license
def test_separate_detect_and_decode_linear_codes(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()

    # Run detection only
    detection_results = detector.detect_linear_codes(frame_2d)
    assert isinstance(detection_results, list)
    for detection_result in detection_results:
        assert isinstance(detection_result, LinearBarcodeDetectionResult)
        _check_barcode_detection_result(detection_result)

    # Run decoding based on the detection results
    decoding_results = detector.decode_linear_codes(detection_results)
    assert isinstance(decoding_results, list)
    assert len(decoding_results) == len(detection_results)
    for decoding_result in decoding_results:
        if decoding_result is not None:
            assert isinstance(decoding_result, LinearBarcodeDecodingResult)
            _check_barcode_decoding_result(decoding_result)

    decoding_results_not_none = [res for res in decoding_results if res is not None]

    # For reference, run combined detection and decoding
    decoding_results_combined = detector.read_linear_codes(frame_2d)

    # Not-None results from decode function should match results from combined function
    assert len(decoding_results_not_none) == len(decoding_results_combined)
    for res_decode, res_combined in zip(decoding_results_not_none, decoding_results_combined, strict=True):
        assert res_decode.code() == res_combined.code()
        assert res_decode.code_format() == res_combined.code_format()
        assert res_decode.center_position() == res_combined.center_position()

    # Try decoding only a subset of detection results
    subset_detection_results = [detection_results[0], detection_results[-1]]
    subset_decoding_results = detector.decode_linear_codes(subset_detection_results)
    assert len(subset_decoding_results) == 2
    for decoding_result in subset_decoding_results:
        if decoding_result is not None:
            assert isinstance(decoding_result, LinearBarcodeDecodingResult)
            _check_barcode_decoding_result(decoding_result)


@pytest.mark.barcode_license
def test_read_matrix_codes(barcodes_frame):
    frame_2d = barcodes_frame.frame_2d()
    detector = BarcodeDetector()
    results = detector.read_matrix_codes(frame_2d)

    assert isinstance(results, list)
    assert len(results) == 15
    for result in results:
        assert isinstance(result, MatrixBarcodeDecodingResult)
        _check_barcode_decoding_result(result)


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
