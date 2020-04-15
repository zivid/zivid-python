# pylint: disable=import-outside-toplevel


def test_calibration_input_init_failure(point_cloud):
    import numpy as np
    import pytest
    import zivid.calibration

    feature_points = zivid.calibration.detect_feature_points(point_cloud)

    elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    data = np.array(elements, dtype=np.float64).reshape((4, 4))
    pose = zivid.calibration.Pose(data)

    with pytest.raises(RuntimeError):
        zivid.calibration.HandEyeInput(pose, feature_points)
