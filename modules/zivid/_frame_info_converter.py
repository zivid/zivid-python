import zivid


def to_frame_info(internal_frame_info):
    def _to_software_version(internal_software_version):

        return zivid.FrameInfo.SoftwareVersion(
            core=internal_software_version.core.value,
        )

    global to_software_version
    to_software_version = _to_software_version
    return zivid.FrameInfo(
        software_version=_to_software_version(internal_frame_info.software_version),
        time_stamp=internal_frame_info.time_stamp.value,
    )
