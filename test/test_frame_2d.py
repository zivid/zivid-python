import numpy as np
import pytest


def test_image_context_manager(frame_2d):
    import zivid

    with frame_2d.image_rgba() as image_rgba:
        assert image_rgba is not None
        assert isinstance(image_rgba, zivid.Image)

    with frame_2d.image_bgra() as image_bgra:
        assert image_bgra is not None
        assert isinstance(image_bgra, zivid.Image)


def test_image(frame_2d):
    import zivid

    image_rgba = frame_2d.image_rgba()
    assert image_rgba is not None
    assert isinstance(image_rgba, zivid.Image)

    image_bgra = frame_2d.image_bgra()
    assert image_bgra is not None
    assert isinstance(image_bgra, zivid.Image)


def test_image_rgba_bgra_correspondence(frame_2d):
    rgba = frame_2d.image_rgba().copy_data()
    bgra = frame_2d.image_bgra().copy_data()

    np.testing.assert_array_equal(bgra[:, :, 0], rgba[:, :, 2])
    np.testing.assert_array_equal(bgra[:, :, 1], rgba[:, :, 1])
    np.testing.assert_array_equal(bgra[:, :, 2], rgba[:, :, 0])
    np.testing.assert_array_equal(bgra[:, :, 3], rgba[:, :, 3])


def test_state(frame_2d):
    import zivid

    state = frame_2d.state
    assert state is not None
    assert isinstance(state, zivid.CameraState)


def test_info(frame_2d):
    import zivid

    info = frame_2d.info
    assert info is not None
    assert isinstance(info, zivid.FrameInfo)


def test_camera_info(frame_2d):
    from zivid.camera_info import CameraInfo

    camera_info = frame_2d.camera_info
    assert camera_info
    assert isinstance(camera_info, CameraInfo)


def test_settings(frame_2d):
    import zivid

    settings_2d = frame_2d.settings
    assert settings_2d is not None
    assert isinstance(settings_2d, zivid.Settings2D)


def test_release(frame_2d):
    frame_2d.image_rgba()
    frame_2d.release()
    with pytest.raises(RuntimeError):
        frame_2d.image_rgba()


def test_context_manager(shared_file_camera):
    import zivid

    settings_2d = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
    with shared_file_camera.capture(settings_2d) as frame_2d:
        frame_2d.image_rgba()
    with pytest.raises(RuntimeError):
        frame_2d.image_rgba()

    with shared_file_camera.capture(settings_2d) as frame_2d:
        frame_2d.image_bgra()
    with pytest.raises(RuntimeError):
        frame_2d.image_bgra()
