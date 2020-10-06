"""Auto generated, do not edit."""
import zivid
import _zivid


def to_camera_info_revision(internal_revision):
    return zivid.CameraInfo.Revision(
        major=internal_revision.major.value, minor=internal_revision.minor.value,
    )


def to_camera_info_user_data(internal_user_data):
    return zivid.CameraInfo.UserData(
        max_size_bytes=internal_user_data.max_size_bytes.value,
    )


def to_camera_info(internal_camera_info):
    return zivid.CameraInfo(
        revision=to_camera_info_revision(internal_camera_info.revision),
        user_data=to_camera_info_user_data(internal_camera_info.user_data),
        firmware_version=internal_camera_info.firmware_version.value,
        model_name=internal_camera_info.model_name.value,
        serial_number=internal_camera_info.serial_number.value,
    )


def to_internal_camera_info_revision(revision):
    internal_revision = _zivid.CameraInfo.Revision()

    internal_revision.major = _zivid.CameraInfo.Revision.Major(revision.major)
    internal_revision.minor = _zivid.CameraInfo.Revision.Minor(revision.minor)

    return internal_revision


def to_internal_camera_info_user_data(user_data):
    internal_user_data = _zivid.CameraInfo.UserData()

    internal_user_data.max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(
        user_data.max_size_bytes
    )

    return internal_user_data


def to_internal_camera_info(camera_info):
    internal_camera_info = _zivid.CameraInfo()

    internal_camera_info.firmware_version = _zivid.CameraInfo.FirmwareVersion(
        camera_info.firmware_version
    )
    internal_camera_info.model_name = _zivid.CameraInfo.ModelName(
        camera_info.model_name
    )
    internal_camera_info.serial_number = _zivid.CameraInfo.SerialNumber(
        camera_info.serial_number
    )

    internal_camera_info.revision = to_internal_camera_info_revision(
        camera_info.revision
    )
    internal_camera_info.user_data = to_internal_camera_info_user_data(
        camera_info.user_data
    )
    return internal_camera_info
