import pytest


def test_init_with():
    import zivid

    with zivid.Application() as app:
        assert app
        assert isinstance(app, zivid.Application)


def test_release(application, sample_data_file):
    assert application.create_file_camera(sample_data_file)
    application.release()
    with pytest.raises(RuntimeError):
        application.create_file_camera(sample_data_file)


def test_create_file_camera(application, sample_data_file):
    import zivid

    file_camera = application.create_file_camera(sample_data_file)
    assert file_camera
    assert isinstance(file_camera, zivid.camera.Camera)


def test_create_file_camera_with_settings(application, sample_data_file):
    import zivid

    with application.create_file_camera(sample_data_file) as file_camera:
        settings = file_camera.settings

    file_camera = application.create_file_camera(sample_data_file, settings=settings)
    assert file_camera
    assert isinstance(file_camera, zivid.camera.Camera)
    assert file_camera.settings == settings


@pytest.mark.physical_camera
def test_connect_camera(application):
    import zivid

    cam = application.connect_camera()
    assert cam
    assert isinstance(cam, zivid.camera.Camera)
    cam.release()


@pytest.mark.physical_camera
def test_connect_camera_serial_number(application):
    import zivid

    with application.connect_camera() as cam:
        serial_number = cam.serial_number

    with application.connect_camera(serial_number) as cam:
        assert cam
        assert isinstance(cam, zivid.camera.Camera)


@pytest.mark.physical_camera
def test_connect_camera_serial_number_and_settings(application):
    import zivid

    with application.connect_camera() as cam:
        serial_number = cam.serial_number

    settings = zivid.Settings()

    with application.connect_camera(
        serial_number=serial_number, settings=settings
    ) as cam:
        assert cam
        assert isinstance(cam, zivid.camera.Camera)
        assert cam.settings == settings


@pytest.mark.physical_camera
def test_connect_camera_settings(application):
    import zivid

    settings = zivid.Settings()
    with application.connect_camera(settings=settings) as cam:
        assert cam
        assert isinstance(cam, zivid.camera.Camera)
        assert cam.settings == settings


def test_cameras_list_of_cameras(application):
    import zivid

    cameras = application.cameras()
    assert isinstance(cameras, list)
    for camera in cameras:
        assert isinstance(camera, zivid.Camera)


def test_cameras_one_camera(application, sample_data_file):
    orig_len = len(application.cameras())
    with application.create_file_camera(sample_data_file) as file_camera:
        assert file_camera
        cameras = application.cameras()
        assert len(cameras) == orig_len + 1
        assert file_camera in cameras


def test_to_string(application):
    string = str(application)
    assert string
    assert isinstance(string, str)
