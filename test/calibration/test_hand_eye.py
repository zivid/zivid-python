import tempfile
from pathlib import Path

import zivid
import numpy as np
import pytest

def test_handeye_input_init_failure(checkerboard_frames, transform):
    frame = checkerboard_frames[0]
    detection_result = zivid.calibration.detect_feature_points(frame.point_cloud())

    with pytest.raises(TypeError):
        # Should fail because pose should come as a Pose, not ndarray
        zivid.calibration.HandEyeInput(transform, detection_result)


def test_handeye_input(checkerboard_frames, transform):
    point_cloud = checkerboard_frames[0].point_cloud()
    detection_result = zivid.calibration.detect_feature_points(point_cloud)
    pose = zivid.calibration.Pose(transform)

    # Check construction of HandEyeInput
    handeye_input = zivid.calibration.HandEyeInput(pose, detection_result)
    assert handeye_input is not None
    assert isinstance(handeye_input, zivid.calibration.HandEyeInput)
    assert str(handeye_input)

    # Check returned Pose
    pose_returned = handeye_input.robot_pose()
    assert pose_returned is not None
    assert isinstance(pose_returned, zivid.calibration.Pose)
    np.testing.assert_array_equal(pose_returned.to_matrix(), transform)

    # Check returned DetectionResult
    detection_result_returned = handeye_input.detection_result()
    assert detection_result_returned is not None
    assert isinstance(detection_result_returned, zivid.calibration.DetectionResult)
    assert detection_result_returned.valid() == detection_result.valid()


def _check_handeye_output(inputs, handeye_output, expected_transform):
    assert isinstance(handeye_output, zivid.calibration.HandEyeOutput)
    assert handeye_output.valid()
    assert bool(handeye_output)
    assert str(handeye_output)

    # Check returned transform
    transform_returned = handeye_output.transform()
    assert isinstance(transform_returned, np.ndarray)
    assert transform_returned.shape == (4, 4)
    np.testing.assert_allclose(transform_returned, expected_transform, rtol=1e-5)

    # Check returned residuals
    residuals_returned = handeye_output.residuals()
    assert isinstance(residuals_returned, list)
    assert len(residuals_returned) == len(inputs)
    for residual in residuals_returned:
        assert isinstance(residual, zivid.calibration.HandEyeResidual)
        assert str(residual)
        assert isinstance(residual.translation(), float)
        assert residual.translation() >= 0.0
        assert isinstance(residual.rotation(), float)
        assert residual.rotation() >= 0.0


def test_eyetohand_calibration(
    handeye_eth_frames, handeye_eth_poses, handeye_eth_transform
):
    # Assemble input
    inputs = []
    for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses):
        inputs.append(
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_feature_points(frame.point_cloud()),
            )
        )

    # Perform eye-to-hand calibration
    handeye_output = zivid.calibration.calibrate_eye_to_hand(inputs)
    _check_handeye_output(inputs, handeye_output, handeye_eth_transform)


def test_marker_eyetohand_calibration(handeye_eth_frames, handeye_eth_poses, handeye_marker_eth_transform):
    # Assemble input
    inputs = []
    for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses):
        inputs.append(
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_markers(frame, [1, 2, 3, 4], zivid.calibration.MarkerDictionary.aruco4x4_50),
            )
        )

    # Perform eye-to-hand calibration
    handeye_output = zivid.calibration.calibrate_eye_to_hand(inputs)
    _check_handeye_output(inputs, handeye_output, handeye_marker_eth_transform)


def test_eyetohand_calibration_save_load(handeye_eth_frames, handeye_eth_poses):
    handeye_output = zivid.calibration.calibrate_eye_to_hand(
        [
            zivid.calibration.HandEyeInput(
                zivid.calibration.Pose(pose_matrix),
                zivid.calibration.detect_feature_points(frame.point_cloud()),
            )
            for frame, pose_matrix in zip(handeye_eth_frames, handeye_eth_poses)
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        transform = handeye_output.transform()
        file_path = Path(tmpdir) / "matrix.yml"
        zivid.Matrix4x4(transform).save(file_path)
        np.testing.assert_allclose(
            zivid.Matrix4x4(transform), zivid.Matrix4x4(file_path), rtol=1e-6
        )