import pytest


def test_is_up_to_date(file_camera):
    from zivid.firmware import is_up_to_date

    assert is_up_to_date(file_camera)


def test_update_without_callback(file_camera):
    import _zivid
    from zivid.firmware import update

    file_camera.disconnect()
    with pytest.raises(RuntimeError):
        update(file_camera)


def test_update_with_progress_callback(file_camera):
    import _zivid
    from zivid.firmware import update

    def callback(progress, description):
        print("{}% {}".format(progress, description), flush=True)

    file_camera.disconnect()

    with pytest.raises(RuntimeError):
        update(file_camera, progress_callback=callback)


def test_update_illegal_callback(file_camera):
    from zivid.firmware import update

    def illegal_callback():
        pass

    file_camera.disconnect()

    with pytest.raises(TypeError):
        update(file_camera, illegal_callback)
