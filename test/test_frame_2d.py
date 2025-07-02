import copy

import numpy as np
import pytest
import zivid
from zivid.camera_info import CameraInfo


def test_image_context_manager(frame_2d):
    with frame_2d.image_rgba() as image_rgba:
        assert image_rgba is not None
        assert isinstance(image_rgba, zivid.Image)

    with frame_2d.image_bgra() as image_bgra:
        assert image_bgra is not None
        assert isinstance(image_bgra, zivid.Image)


def test_image(frame_2d):
    image_rgba = frame_2d.image_rgba()
    assert image_rgba is not None
    assert isinstance(image_rgba, zivid.Image)

    image_bgra = frame_2d.image_bgra()
    assert image_bgra is not None
    assert isinstance(image_bgra, zivid.Image)

    image_rgba_srgb = frame_2d.image_rgba_srgb()
    assert image_rgba_srgb is not None
    assert isinstance(image_rgba_srgb, zivid.Image)

    image_bgra_srgb = frame_2d.image_bgra_srgb()
    assert image_bgra_srgb is not None
    assert isinstance(image_bgra_srgb, zivid.Image)

    image_srgb = frame_2d.image_srgb()
    assert image_srgb is not None
    assert isinstance(image_srgb, zivid.Image)


def test_deprecated_srgb(frame_2d):
    image_rgba_srgb = frame_2d.image_rgba_srgb()
    image_srgb = frame_2d.image_srgb()
    np.testing.assert_array_equal(image_rgba_srgb.copy_data(), image_srgb.copy_data())


def test_image_rgba_bgra_correspondence(frame_2d):
    rgba_linear = frame_2d.image_rgba().copy_data()
    bgra_linear = frame_2d.image_bgra().copy_data()

    rgba_srgb = frame_2d.image_rgba_srgb().copy_data()
    bgra_srgb = frame_2d.image_bgra_srgb().copy_data()

    for rgba, bgra in [(rgba_linear, bgra_linear), (rgba_srgb, bgra_srgb)]:
        np.testing.assert_array_equal(bgra[:, :, 0], rgba[:, :, 2])
        np.testing.assert_array_equal(bgra[:, :, 1], rgba[:, :, 1])
        np.testing.assert_array_equal(bgra[:, :, 2], rgba[:, :, 0])
        np.testing.assert_array_equal(bgra[:, :, 3], rgba[:, :, 3])


def test_state(frame_2d):
    state = frame_2d.state
    assert state is not None
    assert isinstance(state, zivid.CameraState)


def test_info(frame_2d):
    info = frame_2d.info
    assert info is not None
    assert isinstance(info, zivid.FrameInfo)


def test_camera_info(frame_2d):
    camera_info = frame_2d.camera_info
    assert camera_info
    assert isinstance(camera_info, CameraInfo)


def test_settings(frame_2d):
    settings_2d = frame_2d.settings
    assert settings_2d is not None
    assert isinstance(settings_2d, zivid.Settings2D)


def test_release(frame_2d):
    frame_2d.image_rgba()
    frame_2d.release()
    with pytest.raises(RuntimeError):
        frame_2d.image_rgba()


def test_context_manager(shared_file_camera):
    settings_2d = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
    with shared_file_camera.capture(settings_2d) as frame_2d:
        frame_2d.image_rgba()
    with pytest.raises(RuntimeError):
        frame_2d.image_rgba()

    with shared_file_camera.capture(settings_2d) as frame_2d:
        frame_2d.image_bgra()
    with pytest.raises(RuntimeError):
        frame_2d.image_bgra()


def test_copy(frame_2d):
    with copy.copy(frame_2d) as frame_2d_copy:
        assert frame_2d_copy
        assert frame_2d_copy is not frame_2d
        assert isinstance(frame_2d_copy, zivid.Frame2D)
        np.testing.assert_array_equal(frame_2d.image_rgba().copy_data(), frame_2d_copy.image_rgba().copy_data())


def test_deepcopy(frame_2d):
    with copy.deepcopy(frame_2d) as frame_2d_copy:
        assert frame_2d_copy
        assert frame_2d_copy is not frame_2d
        assert isinstance(frame_2d_copy, zivid.Frame2D)
        np.testing.assert_array_equal(frame_2d.image_rgba().copy_data(), frame_2d_copy.image_rgba().copy_data())


def test_clone(frame_2d):
    frame_2d_clone = frame_2d.clone()
    assert frame_2d_clone
    assert frame_2d_clone is not frame_2d
    assert isinstance(frame_2d_clone, zivid.Frame2D)
    np.testing.assert_array_equal(frame_2d.image_rgba().copy_data(), frame_2d_clone.image_rgba().copy_data())
