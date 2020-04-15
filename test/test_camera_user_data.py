import pytest


@pytest.mark.physical_camera
def test_write_user_data(physical_camera):
    physical_camera.write_user_data(b"This is my data")


def test_write_invalid_user_data(file_camera):
    with pytest.raises(TypeError):
        file_camera.write_user_data("This is my data")
    with pytest.raises(TypeError):
        file_camera.write_user_data(1)
    with pytest.raises(TypeError):
        file_camera.write_user_data([1, 2, 3])


@pytest.mark.physical_camera
def test_read_user_data(physical_camera):
    assert isinstance(physical_camera.user_data, bytes)
