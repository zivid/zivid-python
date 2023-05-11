import pytest


@pytest.mark.physical_camera
def test_copy_data(physical_camera_image_2d_bgra):
    import numpy as np

    image = physical_camera_image_2d_bgra
    bgra = image.copy_data()
    assert bgra is not None
    assert isinstance(bgra, np.ndarray)
    assert bgra.shape == (image.height, image.width, 4)
    assert bgra.dtype == np.uint8


@pytest.mark.physical_camera
def test_save_path(physical_camera_image_2d_bgra):
    from pathlib import Path

    physical_camera_image_2d_bgra.save(Path("some_file.png"))


@pytest.mark.physical_camera
def test_save_string(physical_camera_image_2d_bgra):
    physical_camera_image_2d_bgra.save("some_file.png")


@pytest.mark.physical_camera
def test_to_array_context_manager(physical_camera_frame_2d):
    with physical_camera_frame_2d.image_bgra() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


@pytest.mark.physical_camera
def test_save_context_manager(physical_camera_frame_2d):
    with physical_camera_frame_2d.image_bgra() as image_2d:
        image_2d.save("some_file.png")
    with pytest.raises(RuntimeError):
        image_2d.save("some_file.png")


@pytest.mark.physical_camera
def test_height(physical_camera_image_2d_bgra):
    height = physical_camera_image_2d_bgra.height

    assert height is not None
    assert isinstance(height, int)


@pytest.mark.physical_camera
def test_width(physical_camera_image_2d_bgra):
    width = physical_camera_image_2d_bgra.width

    assert width is not None
    assert isinstance(width, int)
