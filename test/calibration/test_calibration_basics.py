def test_pose(transform):
    import numpy as np
    import zivid.calibration

    pose = zivid.calibration.Pose(transform)

    assert pose is not None
    assert isinstance(pose, zivid.calibration.Pose)
    assert str(pose)
    np.testing.assert_array_equal(transform, pose.to_matrix())


def test_detect_feature_points(checkerboard_frames):
    import zivid

    frame = checkerboard_frames[0]
    detection_result = zivid.calibration.detect_feature_points(frame.point_cloud())
    assert detection_result is not None
    assert isinstance(detection_result, zivid.calibration.DetectionResult)
    assert bool(detection_result)
    assert detection_result.valid()
    assert str(detection_result)
