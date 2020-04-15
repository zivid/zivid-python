"""Auto generated, do not edit."""
import zivid
import _zivid


def to_frame_info_software_version(internal_software_version):
    return zivid.FrameInfo.SoftwareVersion(core=internal_software_version.core.value,)


def to_frame_info(internal_frame_info):
    return zivid.FrameInfo(
        software_version=to_frame_info_software_version(
            internal_frame_info.software_version
        ),
        time_stamp=internal_frame_info.time_stamp.value,
    )


def to_internal_frame_info_software_version(software_version):
    internal_software_version = _zivid.FrameInfo.SoftwareVersion()

    internal_software_version.core = _zivid.FrameInfo.SoftwareVersion.Core(
        software_version.core
    )

    return internal_software_version


def to_internal_frame_info(frame_info):
    internal_frame_info = _zivid.FrameInfo()

    internal_frame_info.time_stamp = _zivid.FrameInfo.TimeStamp(frame_info.time_stamp)

    internal_frame_info.software_version = to_internal_frame_info_software_version(
        frame_info.software_version
    )
    return internal_frame_info
