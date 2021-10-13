"""Auto generated, do not edit."""
# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring
import datetime
import _zivid


class FrameInfo:
    class SoftwareVersion:
        def __init__(
            self,
            core=_zivid.FrameInfo.SoftwareVersion.Core().value,
        ):

            if isinstance(core, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(core)
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(
                        value_type=type(core)
                    )
                )

        @property
        def core(self):
            return self._core.value

        @core.setter
        def core(self, value):
            if isinstance(value, (str,)):
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._core == other._core:
                return True
            return False

        def __str__(self):
            return str(_to_internal_frame_info_software_version(self))

    def __init__(
        self,
        time_stamp=_zivid.FrameInfo.TimeStamp().value,
        software_version=None,
    ):

        if isinstance(time_stamp, (datetime.datetime,)):
            self._time_stamp = _zivid.FrameInfo.TimeStamp(time_stamp)
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.datetime,), got {value_type}".format(
                    value_type=type(time_stamp)
                )
            )

        if software_version is None:
            software_version = self.SoftwareVersion()
        if not isinstance(software_version, self.SoftwareVersion):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(software_version))
            )
        self._software_version = software_version

    @property
    def time_stamp(self):
        return self._time_stamp.value

    @property
    def software_version(self):
        return self._software_version

    @time_stamp.setter
    def time_stamp(self, value):
        if isinstance(value, (datetime.datetime,)):
            self._time_stamp = _zivid.FrameInfo.TimeStamp(value)
        else:
            raise TypeError(
                "Unsupported type, expected: datetime.datetime, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @software_version.setter
    def software_version(self, value):
        if not isinstance(value, self.SoftwareVersion):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._software_version = value

    def __eq__(self, other):
        if (
            self._time_stamp == other._time_stamp
            and self._software_version == other._software_version
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_frame_info(self))


def _to_frame_info_software_version(internal_software_version):
    return FrameInfo.SoftwareVersion(
        core=internal_software_version.core.value,
    )


def _to_frame_info(internal_frame_info):
    return FrameInfo(
        software_version=_to_frame_info_software_version(
            internal_frame_info.software_version
        ),
        time_stamp=internal_frame_info.time_stamp.value,
    )


def _to_internal_frame_info_software_version(software_version):
    internal_software_version = _zivid.FrameInfo.SoftwareVersion()

    internal_software_version.core = _zivid.FrameInfo.SoftwareVersion.Core(
        software_version.core
    )

    return internal_software_version


def _to_internal_frame_info(frame_info):
    internal_frame_info = _zivid.FrameInfo()

    internal_frame_info.time_stamp = _zivid.FrameInfo.TimeStamp(frame_info.time_stamp)

    internal_frame_info.software_version = _to_internal_frame_info_software_version(
        frame_info.software_version
    )
    return internal_frame_info
