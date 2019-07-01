import pytest


def test_max_user_data_size(file_camera):
    import numbers

    max_size = file_camera.user_data_max_size_bytes
    assert max_size is not None
    assert isinstance(max_size, numbers.Real)


@pytest.mark.physical_camera
def test_write_user_data(camera):
    camera.write_user_data(b"This is my data")


def test_write_invalid_user_data(file_camera):
    with pytest.raises(TypeError):
        file_camera.write_user_data("This is my data")
    with pytest.raises(TypeError):
        file_camera.write_user_data(1)
    with pytest.raises(TypeError):
        file_camera.write_user_data([1, 2, 3])


@pytest.mark.physical_camera
def test_read_user_data(camera):
    assert isinstance(camera.user_data, bytes)
