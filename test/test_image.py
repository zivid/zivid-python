import copy
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zivid


def test_copy_data(image_2d):
    data = image_2d.copy_data()
    assert data is not None
    assert isinstance(data, np.ndarray)
    assert data.shape == (image_2d.height, image_2d.width, 4)
    assert data.dtype == np.uint8


def test_save_path(image_2d):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_file = Path(tmpdir) / "some_file.png"
        image_2d.save(image_file)


def test_save_string(image_2d):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_file = str(Path(tmpdir) / "some_file.png")
        image_2d.save(image_file)


def test_to_array_context_manager(frame_2d, color_format):
    with getattr(frame_2d, f"image_{color_format}")() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


def test_save_context_manager(frame_2d, color_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_file = str(Path(tmpdir) / "some_file.png")
        with getattr(frame_2d, f"image_{color_format}")() as image_2d:
            image_2d.save(image_file)
        with pytest.raises(RuntimeError):
            image_2d.save(image_file)


def test_height(image_2d):
    height = image_2d.height

    assert height is not None
    assert isinstance(height, int)


def test_width(image_2d):
    width = image_2d.width

    assert width is not None
    assert isinstance(width, int)


def test_load(image_2d, color_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_file = Path(tmpdir) / "saved_image.png"
        image_2d.save(image_file)

        loaded_image = zivid.Image.load(image_file, color_format=color_format)
        np.testing.assert_array_equal(image_2d.copy_data(), loaded_image.copy_data())


def test_load_invalid_color_format(frame_2d):
    with tempfile.TemporaryDirectory() as tmpdir:
        image_file = Path(tmpdir) / "saved_image.png"
        frame_2d.image_rgba().save(image_file)
        with pytest.raises(ValueError):
            zivid.Image.load(image_file, "bgr")
        with pytest.raises(ValueError):
            zivid.Image.load(image_file, "rgb")
        with pytest.raises(ValueError):
            zivid.Image.load(image_file, "asdf")


def test_copy(frame_2d, color_format):
    with getattr(frame_2d, f"image_{color_format}")() as image_2d:
        with copy.copy(image_2d) as copied_image:
            assert copied_image is not None
            assert isinstance(copied_image, zivid.Image)
            assert copied_image is not image_2d
            np.testing.assert_array_equal(image_2d.copy_data(), copied_image.copy_data())
