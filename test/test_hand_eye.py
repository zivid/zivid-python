# pylint: disable=import-outside-toplevel


def test_init_pose():
    import numpy as np
    import zivid.hand_eye

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    data = np.array(elements, dtype=np.float64).reshape((4, 4))

    pose = zivid.hand_eye.Pose(data)

    assert pose is not None
    assert isinstance(pose, zivid.hand_eye.Pose)


def test_detect_feature_points(point_cloud):
    import zivid

    feature_points = zivid.hand_eye.detect_feature_points(point_cloud)

    assert feature_points is not None
    assert isinstance(feature_points, zivid.hand_eye.DetectionResult)


def test_calibration_input_init_failure(point_cloud):
    import numpy as np
    import pytest
    import zivid.hand_eye

    feature_points = zivid.hand_eye.detect_feature_points(point_cloud)

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    data = np.array(elements, dtype=np.float64).reshape((4, 4))

    pose = zivid.hand_eye.Pose(data)

    with pytest.raises(RuntimeError):
        zivid.hand_eye.CalibrationInput(pose, feature_points)
