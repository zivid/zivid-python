import pytest


def test_copy_data(image_2d_rgba):
    import numpy as np

    image = image_2d_rgba
    rgba = image.copy_data()
    assert rgba is not None
    assert isinstance(rgba, np.ndarray)
    assert rgba.shape == (image.height, image.width, 4)
    assert rgba.dtype == np.uint8


def test_save_path(image_2d_rgba):
    from pathlib import Path

    image_2d_rgba.save(Path("some_file.png"))


def test_save_string(image_2d_rgba):
    image_2d_rgba.save("some_file.png")


def test_to_array_context_manager(frame_2d):
    with frame_2d.image_rgba() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


def test_save_context_manager(frame_2d):
    with frame_2d.image_rgba() as image_2d:
        image_2d.save("some_file.png")
    with pytest.raises(RuntimeError):
        image_2d.save("some_file.png")


def test_height(image_2d_rgba):
    height = image_2d_rgba.height

    assert height is not None
    assert isinstance(height, int)


def test_width(image_2d_rgba):
    width = image_2d_rgba.width

    assert width is not None
    assert isinstance(width, int)
