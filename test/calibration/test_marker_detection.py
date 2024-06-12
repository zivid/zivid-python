import zivid
import numpy as np


def _check_markers(markers, markers_expected_2d_corners, markers_expected_3d_corners, markers_expected_poses):
    for marker in markers:
        detected_2d_corners = marker.corners_in_pixel_coordinates
        expected_2d_corners = markers_expected_2d_corners[marker.id]
        np.testing.assert_allclose(detected_2d_corners, expected_2d_corners, rtol=1e-4)

        detected_3d_corners = marker.corners_in_camera_coordinates
        expected_3d_corners = markers_expected_3d_corners[marker.id]
        np.testing.assert_allclose(detected_3d_corners, expected_3d_corners, rtol=1e-4)

        detected_pose = marker.pose.to_matrix()
        expected_pose = markers_expected_poses[marker.id]
        np.testing.assert_allclose(detected_pose, expected_pose, rtol=1e-3)


def test_detect_all_markers(calibration_board_and_aruco_markers_frame, markers_2d_corners, markers_3d_corners,
                            markers_poses):
    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame, [1, 2, 3, 4], zivid.calibration.MarkerDictionary.aruco4x4_50
    )
    assert detection_result
    assert detection_result.valid()
    assert [m.id for m in detection_result.detected_markers()] == [1, 2, 3, 4]
    _check_markers(detection_result.detected_markers(), markers_2d_corners, markers_3d_corners, markers_poses)


def test_detect_filtered_markers(calibration_board_and_aruco_markers_frame, markers_2d_corners,
                                 markers_3d_corners, markers_poses):
    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame, [1, 3], zivid.calibration.MarkerDictionary.aruco4x4_50
    )
    assert detection_result
    assert detection_result.valid()
    assert [m.id for m in detection_result.detected_markers()] == [1, 3]
    _check_markers(detection_result.detected_markers(), markers_2d_corners, markers_3d_corners, markers_poses)


def test_detect_with_some_markers_not_present(calibration_board_and_aruco_markers_frame, markers_2d_corners,
                                              markers_3d_corners, markers_poses):
    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame, [1, 2, 4, 5, 6, 7], zivid.calibration.MarkerDictionary.aruco4x4_50
    )
    assert detection_result
    assert detection_result.valid()
    assert [m.id for m in detection_result.detected_markers()] == [1, 2, 4]
    _check_markers(detection_result.detected_markers(), markers_2d_corners, markers_3d_corners, markers_poses)


def test_detect_specifying_different_dictionary(calibration_board_and_aruco_markers_frame, markers_2d_corners,
                                                markers_3d_corners, markers_poses):
    detection_result = zivid.calibration.detect_markers(
        calibration_board_and_aruco_markers_frame, [1, 2, 3, 4, 7, 9, 42],
        zivid.calibration.MarkerDictionary.aruco6x6_250
    )
    assert not detection_result
    assert not detection_result.valid()
    assert len(detection_result.detected_markers()) == 0

