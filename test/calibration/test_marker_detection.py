import zivid
import numpy as np


def _check_markers(
    markers,
    markers_expected_2d_corners,
    markers_expected_3d_corners,
    markers_expected_poses,
):
    for marker in markers:
        assert isinstance(marker, zivid.calibration.MarkerShape)

        detected_2d_corners = marker.corners_in_pixel_coordinates
        expected_2d_corners = markers_expected_2d_corners[marker.identifier]
        assert isinstance(detected_2d_corners, np.ndarray)
        assert detected_2d_corners.shape == (4, 2)
        np.testing.assert_allclose(detected_2d_corners, expected_2d_corners, rtol=1e-4)

        detected_3d_corners = marker.corners_in_camera_coordinates
        expected_3d_corners = markers_expected_3d_corners[marker.identifier]
        assert isinstance(detected_3d_corners, np.ndarray)
        assert detected_3d_corners.shape == (4, 3)
        np.testing.assert_allclose(detected_3d_corners, expected_3d_corners, rtol=1e-4)

        detected_pose = marker.pose.to_matrix()
        expected_pose = markers_expected_poses[marker.identifier]
        assert isinstance(detected_pose, np.ndarray)
        assert detected_pose.shape == (4, 4)
        np.testing.assert_allclose(detected_pose, expected_pose, rtol=1e-3)


def test_detect_all_markers(
    calibration_board_and_aruco_markers_frame,
    markers_2d_corners,
    markers_3d_corners,
    markers_poses,
):
    allowed_marker_ids = [1, 2, 3, 4]

    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame,
        allowed_marker_ids,
        zivid.calibration.MarkerDictionary.aruco4x4_50,
    )
    assert isinstance(
        detection_result, zivid.calibration.DetectionResultFiducialMarkers
    )
    assert detection_result
    assert detection_result.valid()
    assert detection_result.allowed_marker_ids() == allowed_marker_ids
    assert [
        m.identifier for m in detection_result.detected_markers()
    ] == allowed_marker_ids
    _check_markers(
        detection_result.detected_markers(),
        markers_2d_corners,
        markers_3d_corners,
        markers_poses,
    )


def test_detect_filtered_markers(
    calibration_board_and_aruco_markers_frame,
    markers_2d_corners,
    markers_3d_corners,
    markers_poses,
):
    allowed_marker_ids = [1, 3]

    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame,
        allowed_marker_ids,
        zivid.calibration.MarkerDictionary.aruco4x4_50,
    )
    assert isinstance(
        detection_result, zivid.calibration.DetectionResultFiducialMarkers
    )
    assert detection_result
    assert detection_result.valid()
    assert detection_result.allowed_marker_ids() == allowed_marker_ids
    assert [
        m.identifier for m in detection_result.detected_markers()
    ] == allowed_marker_ids
    _check_markers(
        detection_result.detected_markers(),
        markers_2d_corners,
        markers_3d_corners,
        markers_poses,
    )


def test_detect_with_some_markers_not_present(
    calibration_board_and_aruco_markers_frame,
    markers_2d_corners,
    markers_3d_corners,
    markers_poses,
):
    allowed_marker_ids = [1, 2, 4, 5, 6, 7]

    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame,
        allowed_marker_ids,
        zivid.calibration.MarkerDictionary.aruco4x4_50,
    )
    assert isinstance(
        detection_result, zivid.calibration.DetectionResultFiducialMarkers
    )
    assert detection_result
    assert detection_result.valid()
    assert detection_result.allowed_marker_ids() == allowed_marker_ids
    assert [m.identifier for m in detection_result.detected_markers()] == [1, 2, 4]
    _check_markers(
        detection_result.detected_markers(),
        markers_2d_corners,
        markers_3d_corners,
        markers_poses,
    )


def test_detect_specifying_different_dictionary(
    calibration_board_and_aruco_markers_frame,
):
    allowed_marker_ids = [1, 2, 3, 4, 7, 9, 42]

    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame,
        allowed_marker_ids,
        zivid.calibration.MarkerDictionary.aruco6x6_250,
    )
    assert isinstance(
        detection_result, zivid.calibration.DetectionResultFiducialMarkers
    )
    assert not detection_result
    assert not detection_result.valid()
    assert detection_result.allowed_marker_ids() == allowed_marker_ids
    assert len(detection_result.detected_markers()) == 0


def test_marker_dictionary():
    from zivid.calibration import MarkerDictionary

    assert MarkerDictionary.marker_count(MarkerDictionary.aruco4x4_50) == 50
    assert MarkerDictionary.marker_count(MarkerDictionary.aruco6x6_250) == 250

    assert MarkerDictionary.aruco5x5_1000 in MarkerDictionary.valid_values()
