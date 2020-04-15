# pylint: disable=import-outside-toplevel


def test_init_pose():
    import numpy as np
    import zivid.calibration

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    data = np.array(elements, dtype=np.float64).reshape((4, 4))

    pose = zivid.calibration.Pose(data)

    assert pose is not None
    assert isinstance(pose, zivid.calibration.Pose)


def test_detect_feature_points(point_cloud):
    import zivid

    feature_points = zivid.calibration.detect_feature_points(point_cloud)

    assert feature_points is not None
    assert isinstance(feature_points, zivid.calibration.DetectionResult)
