from pathlib import Path
import tempfile
import numpy as np
import pytest
import zivid


def test_copy_data(image_2d_srgb):
    image = image_2d_srgb
    rgba = image.copy_data()
    assert rgba is not None
    assert isinstance(rgba, np.ndarray)
    assert rgba.shape == (image.height, image.width, 4)
    assert rgba.dtype == np.uint8


def test_save_path(image_2d_srgb):
    image_2d_srgb.save(Path("some_file.png"))


def test_save_string(image_2d_srgb):
    image_2d_srgb.save("some_file.png")


def test_to_array_context_manager(frame_2d):
    with frame_2d.image_srgb() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


def test_save_context_manager(frame_2d):
    with frame_2d.image_srgb() as image_2d:
        image_2d.save("some_file.png")
    with pytest.raises(RuntimeError):
        image_2d.save("some_file.png")


def test_height(image_2d_srgb):
    height = image_2d_srgb.height

    assert height is not None
    assert isinstance(height, int)


def test_width(image_2d_srgb):
    width = image_2d_srgb.width

    assert width is not None
    assert isinstance(width, int)


def test_load(frame_2d: zivid.Frame2D):
    with frame_2d.image_srgb() as image_2d:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_file = Path(tmpdir) / "saved_image.png"
            image_2d.save(image_file)

            loaded_image = zivid.Image.load(image_file, "srgb")
            np.testing.assert_array_equal(
                image_2d.copy_data(), loaded_image.copy_data()
            )

            with pytest.raises(ValueError):
                zivid.Image.load(image_file, "srg")
