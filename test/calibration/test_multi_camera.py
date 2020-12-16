def test_multicamera_invalid_input_single_detectionresult(checkerboard_frames):
    import pytest
    import zivid.calibration

    frame = checkerboard_frames[0]
    detection_result = zivid.calibration.detect_feature_points(frame.point_cloud())

    with pytest.raises(TypeError):
        # Will fail because a list of detection_result is required
        zivid.calibration.calibrate_multi_camera(detection_result)

    with pytest.raises(RuntimeError):
        # Will fail because the list must have more than one detection result
        zivid.calibration.calibrate_multi_camera([detection_result])


def test_multicamera_calibration(checkerboard_frames, multicamera_transforms):
    import numpy as np
    import zivid.calibration

    # Detect feature points
    detection_results = [
        zivid.calibration.detect_feature_points(frame.point_cloud())
        for frame in checkerboard_frames
    ]
    assert all(detection_results)

    # Perform multicamera calibration
    multicamera_output = zivid.calibration.calibrate_multi_camera(detection_results)
    assert isinstance(multicamera_output, zivid.calibration.MultiCameraOutput)
    assert str(multicamera_output)
    assert bool(multicamera_output)

    # Extract and check transforms
    transforms = multicamera_output.transforms()
    assert isinstance(transforms, list)
    assert len(transforms) == len(detection_results)
    np.testing.assert_array_equal(transforms[0], np.eye(4))

    for transform, reference in zip(transforms, multicamera_transforms):
        assert isinstance(transform, np.ndarray)
        assert transform.shape == (4, 4)
        np.testing.assert_array_equal(transform[-1, :], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(transform, reference, rtol=1e-4)

    # Extract and check residuals
    residuals = multicamera_output.residuals()
    assert isinstance(residuals, list)
    assert len(residuals) == len(detection_results)
    for i, residual in enumerate(residuals):
        assert isinstance(residual, zivid.calibration.MultiCameraResidual)
        assert str(residual)
        assert isinstance(residual.translation(), float)
        if i == 0:
            assert residual.translation() == 0.0
        else:
            assert residual.translation() > 0.0
