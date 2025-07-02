import numpy as np
import pytest
import zivid
import zivid.experimental.hand_eye_low_dof


def test_fixed_placement_of_fiducial_marker():
    marker = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2, 3])
    assert str(marker) == "{ id: 0, position: { x: 1, y: 2, z: 3 } }"
    assert marker.position == [1, 2, 3]
    assert marker.id == 0

    marker = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(1, [1.5, 2.5, 3.5])
    assert str(marker) == "{ id: 1, position: { x: 1.5, y: 2.5, z: 3.5 } }"
    marker = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(2, (10, 10.5, 11))
    assert str(marker) == "{ id: 2, position: { x: 10, y: 10.5, z: 11 } }"

    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, 1)
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [])
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1])
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2])
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2, 3, 4])
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2, "string"])


def test_fixed_placement_of_fiducial_markers():
    marker1 = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2, 3])
    marker2 = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(1, [4, 5, 6])
    markers = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarkers(
        zivid.calibration.MarkerDictionary.aruco4x4_50, [marker1, marker2]
    )

    assert str(markers) == (
        "{ dictionary: aruco4x4_50, markers: { { id: 0, position: { x: 1, y: 2, z: 3 "
        "} }, { id: 1, position: { x: 4, y: 5, z: 6 } } } }"
    )

    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarkers(
            zivid.calibration.MarkerDictionary.aruco4x4_50, [1, 2, 3]
        )
    with pytest.raises(ValueError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarkers(
            "invalid_marker_dictionary", [marker1, marker2]
        )


def test_fixed_placement_of_calibration_board():
    calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard([1, 2, 3])
    assert str(calibration_board) == "point: { x: 1, y: 2, z: 3 }"
    calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard((1.5, 2, 3.5))
    assert str(calibration_board) == "point: { x: 1.5, y: 2, z: 3.5 }"

    calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard(
        zivid.calibration.Pose(zivid.Matrix4x4.identity())
    )
    assert str(calibration_board) == (
        "pose: [ [ 1.000000,  0.000000,  0.000000,  0.000000], \n"
        "  [ 0.000000,  1.000000,  0.000000,  0.000000], \n"
        "  [ 0.000000,  0.000000,  1.000000,  0.000000], \n"
        "  [ 0.000000,  0.000000,  0.000000,  1.000000] ]"
    )

    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard([1, 2, "string"])
    with pytest.raises(TypeError):
        zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard(1)


def test_fixed_placement_of_calibration_objects():
    calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard([1, 2, 3])
    calibration_objects = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationObjects(calibration_board)
    assert str(calibration_objects) == "calibrationBoard: point: { x: 1, y: 2, z: 3 }"

    marker = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(0, [1, 2, 3])
    markers = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarkers(
        zivid.calibration.MarkerDictionary.aruco4x4_250, [marker]
    )
    calibration_objects = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationObjects(markers)
    assert (
        str(calibration_objects)
        == "markers: { dictionary: aruco4x4_250, markers: { { id: 0, position: { x: 1, y: 2, z: 3 } } } }"
    )


def test_eth_transform_low_dof_approximate_match(
    handeye_eth_transform,
    handeye_marker_eth_transform,
    handeye_eth_low_dof_transform,
    handeye_eth_low_dof_markers_transform,
):
    # Ensure that the low DOF calibration transforms are approximately the same as their
    # full 6-DOF calibration transform counterparts.
    np.testing.assert_allclose(handeye_eth_transform, handeye_eth_low_dof_transform, rtol=2.5e-2)
    np.testing.assert_allclose(handeye_marker_eth_transform, handeye_eth_low_dof_markers_transform, rtol=2.5e-2)


def test_eye_to_hand_low_dof_calibration_with_calibration_board(
    handeye_eth_frames,
    handeye_eth_poses,
    handeye_eth_low_dof_fixed_calibration_board_pose,
    handeye_eth_low_dof_transform,
):
    inputs = []
    for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses):
        inputs.append(
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_calibration_board(frame),
            )
        )

    calibration_board_pose = zivid.calibration.Pose(handeye_eth_low_dof_fixed_calibration_board_pose)

    fixed_calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard(
        calibration_board_pose
    )
    fixed_objects = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationObjects(fixed_calibration_board)

    handeye_output = zivid.experimental.hand_eye_low_dof.calibrate_eye_to_hand_low_dof(inputs, fixed_objects)
    pytest.helpers.check_handeye_output(inputs, handeye_output, handeye_eth_low_dof_transform)


def test_eye_to_hand_low_dof_calibration_with_markers(
    handeye_eth_frames,
    handeye_eth_poses,
    handeye_eth_low_dof_markers_transform,
    handeye_eth_low_dof_fixed_markers_id_position_list,
):
    inputs = []
    for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses):
        inputs.append(
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_markers(frame, [1, 2, 3, 4], zivid.calibration.MarkerDictionary.aruco4x4_50),
            )
        )

    fixed_marker_list = []
    for marker_id, position in handeye_eth_low_dof_fixed_markers_id_position_list:
        fixed_marker_list.append(
            zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarker(marker_id, position)
        )

    fixed_markers = zivid.experimental.hand_eye_low_dof.FixedPlacementOfFiducialMarkers(
        zivid.calibration.MarkerDictionary.aruco4x4_50, fixed_marker_list
    )
    fixed_objects = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationObjects(fixed_markers)

    handeye_output = zivid.experimental.hand_eye_low_dof.calibrate_eye_to_hand_low_dof(inputs, fixed_objects)
    pytest.helpers.check_handeye_output(inputs, handeye_output, handeye_eth_low_dof_markers_transform)


def test_eye_in_hand_low_dof_calibration_with_eye_to_hand_data(
    handeye_eth_frames,
    handeye_eth_poses,
    handeye_eth_low_dof_fixed_calibration_board_pose,
):
    # This is a negative test, it won't calibrate correctly since we are using
    # eye-to-hand calibration data with the eye-in-hand functionality.
    inputs = []
    for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses):
        inputs.append(
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_calibration_board(frame),
            )
        )

    calibration_board_pose = zivid.calibration.Pose(handeye_eth_low_dof_fixed_calibration_board_pose)
    fixed_calibration_board = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationBoard(
        calibration_board_pose
    )
    fixed_objects = zivid.experimental.hand_eye_low_dof.FixedPlacementOfCalibrationObjects(fixed_calibration_board)

    with pytest.raises(RuntimeError):
        _ = zivid.experimental.hand_eye_low_dof.calibrate_eye_in_hand_low_dof(inputs, fixed_objects)
