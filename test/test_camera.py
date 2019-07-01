import pytest


def test_illegal_init(application):  # pylint: disable=unused-argument
    import zivid

    with pytest.raises(RuntimeError):
        zivid.camera.Camera("this should fail")

    with pytest.raises(RuntimeError):
        zivid.camera.Camera(None)

    with pytest.raises(RuntimeError):
        zivid.camera.Camera(12345)


def test_init_with(application, sample_data_file):
    import zivid

    with application.create_file_camera(sample_data_file) as file_camera:
        assert file_camera
        assert isinstance(file_camera, zivid.camera.Camera)


def test_capture(file_camera):
    import zivid

    frame = file_camera.capture()
    assert frame
    assert isinstance(frame, zivid.frame.Frame)
    frame.release()


def test_get_settings(file_camera):
    import zivid

    settings = file_camera.settings
    assert settings
    assert isinstance(settings, zivid.Settings)


def test_set_settings(file_camera):
    import zivid

    settings = zivid.Settings(
        iris=21  # Must be 21. File camera's default settings are almost similar to
        # the default settings, apart from iris, which is for some reason 21
    )
    file_camera.settings = settings
    assert file_camera.settings == settings
    assert isinstance(file_camera.settings, zivid.Settings)


def test_equal(application, sample_data_file):
    import zivid

    with application.create_file_camera(sample_data_file) as file_camera:
        camera_handle = zivid.Camera(
            file_camera._Camera__impl  # pylint: disable=protected-access
        )
        assert isinstance(file_camera, zivid.Camera)
        assert isinstance(camera_handle, zivid.Camera)
        assert camera_handle == file_camera


def test_not_equal(application, sample_data_file):
    with application.create_file_camera(
        sample_data_file
    ) as file_camera1, application.create_file_camera(sample_data_file) as file_camera2:
        assert file_camera1 != file_camera2


def test_disconnect(file_camera):
    assert file_camera.state.connected
    file_camera.disconnect()
    assert not file_camera.state.connected


def test_connect_no_settings(file_camera):
    file_camera.disconnect()
    assert not file_camera.state.connected
    file_camera.connect()
    assert file_camera.state.connected


def test_connect_with_settings(file_camera):
    settings = file_camera.settings
    # leaving settings untouched since file camera has not correctly implemented set or get settings
    file_camera.disconnect()

    assert not file_camera.state.connected

    file_camera.connect(settings=settings)

    assert file_camera.state.connected
    assert file_camera.settings == settings


def test_to_string(file_camera):
    string = str(file_camera)
    assert string
    assert isinstance(string, str)


def test_update_settings_all_settings(file_camera):
    settings = file_camera.settings
    # leaving settings untouched since file camera has not correctly implemented set or get settings

    with file_camera.update_settings() as updater:
        updater.settings = settings

    assert file_camera.settings == settings


def test_update_settings_one_setting(file_camera):
    iris = file_camera.settings.iris
    # leaving settings untouched since file camera has not correctly implemented set or get settings
    with file_camera.update_settings() as updater:
        updater.settings.iris = iris

    assert file_camera.settings.iris == iris
