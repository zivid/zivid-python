def test_init_pose():
    import numpy as np
    import zivid.handeye

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    data = np.array(elements, dtype=np.float64).reshape((4, 4))

    pose = zivid.handeye.Pose(data)

    assert pose is not None
    assert isinstance(pose, zivid.handeye.Pose)


def test_detect_feature_points(point_cloud):
    import zivid

    feature_points = zivid.handeye.detect_feature_points(point_cloud)

    assert feature_points is not None
    assert isinstance(feature_points, zivid.handeye.DetectionResult)


def test_calibration_input_init_failure(point_cloud):
    import numpy as np
    import pytest
    import zivid.handeye

    feature_points = zivid.handeye.detect_feature_points(point_cloud)

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    data = np.array(elements, dtype=np.float64).reshape((4, 4))

    pose = zivid.handeye.Pose(data)

    with pytest.raises(RuntimeError):
        zivid.handeye.CalibrationInput(pose, feature_points)
