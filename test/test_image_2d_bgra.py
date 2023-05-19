import pytest


def test_copy_data(image_2d_bgra):
    import numpy as np

    image = image_2d_bgra
    bgra = image.copy_data()
    assert bgra is not None
    assert isinstance(bgra, np.ndarray)
    assert bgra.shape == (image.height, image.width, 4)
    assert bgra.dtype == np.uint8


def test_save_path(image_2d_bgra):
    from pathlib import Path

    image_2d_bgra.save(Path("some_file.png"))


def test_save_string(image_2d_bgra):
    image_2d_bgra.save("some_file.png")


def test_to_array_context_manager(frame_2d):
    with frame_2d.image_bgra() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


def test_save_context_manager(frame_2d):
    with frame_2d.image_bgra() as image_2d:
        image_2d.save("some_file.png")
    with pytest.raises(RuntimeError):
        image_2d.save("some_file.png")


def test_height(image_2d_bgra):
    height = image_2d_bgra.height

    assert height is not None
    assert isinstance(height, int)


def test_width(image_2d_bgra):
    width = image_2d_bgra.width

    assert width is not None
    assert isinstance(width, int)
