import pytest
import zivid


def test_create_file_camera(application, file_camera_file):
    file_camera = application.create_file_camera(file_camera_file)
    assert file_camera
    assert isinstance(file_camera, zivid.camera.Camera)


@pytest.mark.physical_camera
def test_connect_camera(application):
    cam = application.connect_camera()
    assert cam
    assert isinstance(cam, zivid.camera.Camera)
    cam.release()


@pytest.mark.physical_camera
def test_connect_camera_serial_number(application):
    with application.connect_camera() as cam:
        serial_number = cam.info.serial_number

    with application.connect_camera(serial_number) as cam:
        assert cam
        assert isinstance(cam, zivid.camera.Camera)


def test_cameras_list_of_cameras(application):
    cameras = application.cameras()
    assert isinstance(cameras, list)
    for camera in cameras:
        assert isinstance(camera, zivid.Camera)


def test_cameras_one_camera(application, file_camera_file):
    orig_len = len(application.cameras())
    with application.create_file_camera(file_camera_file) as file_camera:
        assert file_camera
        cameras = application.cameras()
        assert len(cameras) == orig_len + 1
        assert file_camera in cameras


def test_to_string(application):
    string = str(application)
    assert string
    assert isinstance(string, str)


def test_connect_camera_raises_when_serial_number_and_address_both_given(application):
    with pytest.raises(ValueError):
        application.connect_camera(serial_number="ABC123", address=zivid.CameraAddress("192.168.0.1"))


@pytest.mark.physical_camera
def test_connect_camera_with_address(application):
    with application.connect_camera() as cam:
        ip_address = cam.state.network.ipv4.address

    with application.connect_camera(address=zivid.CameraAddress(ip_address)) as cam:
        assert cam
        assert isinstance(cam, zivid.Camera)
