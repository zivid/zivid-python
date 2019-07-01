import pytest


def test_revision(file_camera):
    import zivid

    revision = file_camera.revision
    assert revision is not None
    assert isinstance(revision, zivid.camera.Camera.Revision)
    assert revision == zivid.camera.Camera.Revision(0, 0)


def test_firmware_version(file_camera):
    firmware_version = file_camera.firmware_version
    assert firmware_version is not None
    assert isinstance(firmware_version, str)
    assert firmware_version == "NA"


def test_model_name(file_camera):
    model_name = file_camera.model_name
    assert model_name is not None
    assert isinstance(model_name, str)
    assert model_name.startswith("FileCamera")


def test_serial_number(file_camera):
    serial_number = file_camera.serial_number
    assert serial_number is not None
    assert isinstance(serial_number, str)
    assert serial_number.startswith("F")


def test_illegal_set_revision(file_camera):
    with pytest.raises(AttributeError):
        file_camera.revision = file_camera.revision


def test_illegal_set_firmware_version(file_camera):
    with pytest.raises(AttributeError):
        file_camera.firmware_version = file_camera.firmware_version


def test_illegal_set_model_name(file_camera):
    with pytest.raises(AttributeError):
        file_camera.model_name = file_camera.model_name


def test_illegal_set_serial_number(file_camera):
    with pytest.raises(AttributeError):
        file_camera.serial_number = file_camera.serial_number
