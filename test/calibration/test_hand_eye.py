# pylint: disable=import-outside-toplevel


def test_handeye_input_init_failure(point_cloud, transform):
    import pytest
    import zivid.calibration

    detection_result = zivid.calibration.detect_feature_points(point_cloud)

    with pytest.raises(TypeError):
        # Should fail because pose should come as a Pose, not ndarray
        zivid.calibration.HandEyeInput(transform, detection_result)

    pose = zivid.calibration.Pose(transform)
    with pytest.raises(RuntimeError):
        # Should fail because point cloud did not include checkerboard
        zivid.calibration.HandEyeInput(pose, detection_result)


def test_handeye_input(checkerboard_frames, transform):

    import numpy as np
    import zivid.calibration

    point_cloud = checkerboard_frames[0].point_cloud()
    detection_result = zivid.calibration.detect_feature_points(point_cloud)
    pose = zivid.calibration.Pose(transform)

    # Check construction of HandEyeInput
    handeye_input = zivid.calibration.HandEyeInput(pose, detection_result)
    assert handeye_input is not None
    assert isinstance(handeye_input, zivid.calibration.HandEyeInput)
    assert str(handeye_input)

    # Check returned Pose
    pose_returned = handeye_input.pose()
    assert pose_returned is not None
    assert isinstance(pose_returned, zivid.calibration.Pose)
    np.testing.assert_array_equal(pose_returned.to_matrix(), transform)

    # Check returned DetectionResult
    detection_result_returned = handeye_input.detection_result()
    assert detection_result_returned is not None
    assert isinstance(detection_result_returned, zivid.calibration.DetectionResult)
    assert detection_result_returned.valid() == detection_result.valid()
