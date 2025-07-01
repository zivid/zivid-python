from zivid import CameraIntrinsics
from zivid.experimental.calibration import estimate_intrinsics, intrinsics
from zivid.settings import Settings
from zivid.settings2d import Settings2D


def _check_camera_intrinsics(camera_intrinsics):
    assert isinstance(camera_intrinsics, CameraIntrinsics)
    assert isinstance(camera_intrinsics.camera_matrix, CameraIntrinsics.CameraMatrix)
    assert isinstance(camera_intrinsics.distortion, CameraIntrinsics.Distortion)

    assert isinstance(camera_intrinsics.camera_matrix.fx, float)
    assert isinstance(camera_intrinsics.camera_matrix.fy, float)
    assert isinstance(camera_intrinsics.camera_matrix.cx, float)
    assert isinstance(camera_intrinsics.camera_matrix.cy, float)

    assert isinstance(camera_intrinsics.distortion.k1, float)
    assert isinstance(camera_intrinsics.distortion.k2, float)
    assert isinstance(camera_intrinsics.distortion.k3, float)
    assert isinstance(camera_intrinsics.distortion.p1, float)
    assert isinstance(camera_intrinsics.distortion.p2, float)


def test_intrinsics(shared_file_camera):
    camera_intrinsics = intrinsics(shared_file_camera)
    _check_camera_intrinsics(camera_intrinsics)


def test_intrinsics_with_settings_2d(shared_file_camera):
    camera_intrinsics = intrinsics(
        camera=shared_file_camera,
        settings=Settings2D(acquisitions=[Settings2D.Acquisition()]),
    )
    _check_camera_intrinsics(camera_intrinsics)


def test_intrinsics_with_settings_3d(shared_file_camera):
    camera_intrinsics = intrinsics(
        camera=shared_file_camera,
        settings=Settings(acquisitions=[Settings.Acquisition()]),
    )
    _check_camera_intrinsics(camera_intrinsics)


def test_estimate_intrinsics(frame):
    camera_intrinsics = estimate_intrinsics(frame)
    _check_camera_intrinsics(camera_intrinsics)
