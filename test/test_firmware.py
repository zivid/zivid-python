import pytest
from zivid.firmware import is_up_to_date, update


def test_is_up_to_date(shared_file_camera):
    assert is_up_to_date(shared_file_camera)


def test_update_without_callback(file_camera):
    file_camera.disconnect()
    with pytest.raises(RuntimeError):
        update(file_camera)


def test_update_with_progress_callback(file_camera):
    def callback(progress, description):
        print("{}% {}".format(progress, description), flush=True)

    file_camera.disconnect()

    with pytest.raises(RuntimeError):
        update(file_camera, progress_callback=callback)


def test_update_illegal_callback(file_camera):
    def illegal_callback():
        pass

    file_camera.disconnect()

    with pytest.raises(TypeError):
        update(file_camera, illegal_callback)
