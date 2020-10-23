import pytest


@pytest.mark.physical_camera
def test_copy_data(physical_camera_image_2d):
    import numpy as np

    image = physical_camera_image_2d
    rgba = image.copy_data()
    assert rgba is not None
    assert isinstance(rgba, np.ndarray)
    assert rgba.shape == (image.height, image.width, 4)
    assert rgba.dtype == np.uint8


@pytest.mark.physical_camera
def test_save_path(physical_camera_image_2d):
    from pathlib import Path

    physical_camera_image_2d.save(Path("some_file.png"))


@pytest.mark.physical_camera
def test_save_string(physical_camera_image_2d):
    physical_camera_image_2d.save("some_file.png")


@pytest.mark.physical_camera
def test_to_array_context_manager(physical_camera_frame_2d):
    with physical_camera_frame_2d.image_rgba() as image_2d:
        image_2d.copy_data()
    with pytest.raises(RuntimeError):
        image_2d.copy_data()


@pytest.mark.physical_camera
def test_save_context_manager(physical_camera_frame_2d):
    with physical_camera_frame_2d.image_rgba() as image_2d:
        image_2d.save("some_file.png")
    with pytest.raises(RuntimeError):
        image_2d.save("some_file.png")


@pytest.mark.physical_camera
def test_height(physical_camera_image_2d):
    height = physical_camera_image_2d.height

    assert height is not None
    assert isinstance(height, int)


@pytest.mark.physical_camera
def test_width(physical_camera_image_2d):
    width = physical_camera_image_2d.width

    assert width is not None
    assert isinstance(width, int)
