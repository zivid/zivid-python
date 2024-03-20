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


def test_calibration_board_pose(calibration_board_frame):
    import numpy as np
    import zivid

    point_cloud = calibration_board_frame.point_cloud()
    detection_result = zivid.calibration.detect_feature_points(point_cloud)
    pose = detection_result.pose()
    assert pose is not None
    assert str(pose)
    assert isinstance(pose, zivid.calibration.Pose)
    np.testing.assert_allclose(
        pose.to_matrix(),
        [
            [0.996286, -0.004492, -0.085990, -194.331055],
            [0.004557, 0.999989, 0.000553, -193.129150],
            [0.085987, -0.000943, 0.996296, 1913.698975],
            [0.000000, 0.000000, 0.000000, 1.000000],
        ],
        rtol=0.0004,
    )
