# pylint: disable=import-outside-toplevel


def test_pose(transform):
    import numpy as np
    import zivid.calibration

    pose = zivid.calibration.Pose(transform)

    assert pose is not None
    assert isinstance(pose, zivid.calibration.Pose)
    assert str(pose)
    np.testing.assert_array_equal(transform, pose.to_matrix())


def test_detect_feature_points(point_cloud):
    import zivid

    feature_points = zivid.calibration.detect_feature_points(point_cloud)

    assert feature_points is not None
    assert isinstance(feature_points, zivid.calibration.DetectionResult)
