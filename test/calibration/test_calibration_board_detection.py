from zivid import calibration


def _check_detection_result(detection_result):
    assert detection_result.valid()
    assert detection_result
    assert len(detection_result.centroid() == 3)
    assert isinstance(detection_result.pose(), calibration.Pose)
    assert detection_result.status() == calibration.CalibrationBoardDetectionStatus.ok
    assert isinstance(detection_result.status_description(), str)


def test_detect_calibration_board_frame(calibration_board_and_aruco_markers_frame):
    detection_result = calibration.detect_calibration_board(
        calibration_board_and_aruco_markers_frame
    )
    _check_detection_result(detection_result)


def test_detect_calibration_board_file_camera(file_camera_calibration_board):
    detection_result = calibration.detect_calibration_board(
        file_camera_calibration_board
    )
    _check_detection_result(detection_result)


def test_detect_calibration_board_invalid_file_camera(shared_file_camera):
    detection_result = calibration.detect_calibration_board(shared_file_camera)
    assert not detection_result.valid()
    assert not detection_result
    assert (
        detection_result.status()
        == calibration.CalibrationBoardDetectionStatus.no_valid_fiducial_marker_detected
    )
    assert isinstance(detection_result.status_description(), str)


def test_capture_calibration_board_and_then_detect(file_camera_calibration_board):
    frame = calibration.capture_calibration_board(file_camera_calibration_board)
    detection_result = calibration.detect_calibration_board(frame)
    _check_detection_result(detection_result)
