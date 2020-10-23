def test_revision(file_camera_info):
    import zivid

    revision = file_camera_info.revision
    assert revision is not None
    assert isinstance(revision, zivid.CameraInfo.Revision)
    assert revision == zivid.CameraInfo.Revision(0, 0)


def test_firmware_version(file_camera_info):
    firmware_version = file_camera_info.firmware_version
    assert firmware_version is not None
    assert isinstance(firmware_version, str)
    assert firmware_version == "NA"


def test_model_name(file_camera_info):
    model_name = file_camera_info.model_name
    assert model_name is not None
    assert isinstance(model_name, str)
    assert model_name.startswith("FileCamera")


def test_serial_number(file_camera_info):
    serial_number = file_camera_info.serial_number
    assert serial_number is not None
    assert isinstance(serial_number, str)
    assert serial_number.startswith("F")


def test_user_data(file_camera_info):
    import zivid

    user_data = file_camera_info.user_data
    assert user_data is not None
    assert isinstance(user_data, zivid.camera_info.CameraInfo.UserData)

    assert user_data.max_size_bytes is not None
    assert isinstance(user_data.max_size_bytes, int)
    assert user_data.max_size_bytes == 0
