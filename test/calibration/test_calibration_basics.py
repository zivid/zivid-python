def test_pose(transform):
    import numpy as np
    import zivid.calibration

    pose = zivid.calibration.Pose(transform)

    assert pose is not None
    assert isinstance(pose, zivid.calibration.Pose)
    assert str(pose)
    np.testing.assert_array_equal(transform, pose.to_matrix())


def test_detect_feature_points(checkerboard_frames):
    import numpy as np
    import zivid

    frame = checkerboard_frames[0]
    detection_result = zivid.calibration.detect_feature_points(frame.point_cloud())
    assert detection_result is not None
    assert isinstance(detection_result, zivid.calibration.DetectionResult)
    assert bool(detection_result)
    assert detection_result.valid()
    assert str(detection_result)
    centroid = detection_result.centroid()
    assert isinstance(centroid, np.ndarray)
    assert centroid.shape == (3,)
    np.testing.assert_allclose(centroid, [-67.03593, 71.17018, 906.348], rtol=1e-6)
