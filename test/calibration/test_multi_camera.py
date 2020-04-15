# pylint: disable=import-outside-toplevel


def test_calibration_dummy(point_cloud):
    import pytest
    import zivid.calibration

    feature_points = zivid.calibration.detect_feature_points(point_cloud)

    with pytest.raises(RuntimeError):
        zivid.calibration.calibrate_multi_camera([feature_points, feature_points])
