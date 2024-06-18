def test_detect_calibration_board_frame(calibration_board_and_aruco_markers_frame):
    from zivid import calibration

    detection_result = calibration.detect_calibration_board(
        calibration_board_and_aruco_markers_frame
    )
    assert detection_result.valid()


def test_detect_calibration_board_file_camera(file_camera_calibration_board):
    from zivid import calibration

    detection_result = calibration.detect_calibration_board(
        file_camera_calibration_board
    )
    assert detection_result.valid()


def test_detect_calibration_board_invalid_file_camera(shared_file_camera):
    from zivid import calibration

    detection_result = calibration.detect_calibration_board(shared_file_camera)
    assert not detection_result.valid()


def test_capture_calibration_board_and_then_detect(file_camera_calibration_board):
    from zivid import calibration

    frame = calibration.capture_calibration_board(file_camera_calibration_board)
    detection_result = calibration.detect_calibration_board(frame)
    assert detection_result.valid()
