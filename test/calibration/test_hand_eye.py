# pylint: disable=import-outside-toplevel


def test_calibration_input_init_failure(point_cloud, transform):
    import pytest
    import zivid.calibration

    feature_points = zivid.calibration.detect_feature_points(point_cloud)

    with pytest.raises(TypeError):
        # Should fail because pose should come as a Pose, not ndarray
        zivid.calibration.HandEyeInput(transform, feature_points)

    pose = zivid.calibration.Pose(transform)
    with pytest.raises(RuntimeError):
        # Should fail because point cloud did not include checkerboard
        zivid.calibration.HandEyeInput(pose, feature_points)
