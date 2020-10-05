"""Auto generated, do not edit."""
# pylint: disable=missing-class-docstring,missing-function-docstring
import datetime
import _zivid
import zivid
import zivid._frame_info_converter


class FrameInfo:
    class SoftwareVersion:
        def __init__(
            self, core=_zivid.FrameInfo().SoftwareVersion().Core().value,
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
            return str(
                zivid._frame_info_converter.to_internal_frame_info_software_version(
                    self
                )
            )

    def __init__(
        self, time_stamp=_zivid.FrameInfo().TimeStamp().value, software_version=None,
    ):

        if isinstance(time_stamp, (datetime.datetime,)):
            if time_stamp < datetime.datetime(1970, 1, 1):
                raise ValueError(
                    "Unsupported time stamp: '{time_stamp}', time stamp can only be set to a time point after January 1st, 1970".format(
                        time_stamp=time_stamp
                    )
                )
            self._time_stamp = _zivid.FrameInfo.TimeStamp(time_stamp)
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.datetime,), got {value_type}".format(
                    value_type=type(time_stamp)
                )
            )
        if software_version is None:
            software_version = zivid.FrameInfo.SoftwareVersion()
        if not isinstance(software_version, zivid.FrameInfo.SoftwareVersion):
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
            if value < datetime.datetime(1970, 1, 1):
                raise ValueError(
                    "Unsupported time stamp: '{value}', time stamp can only be set to a time point after January 1st, 1970".format(
                        value=value
                    )
                )
            self._time_stamp = _zivid.FrameInfo.TimeStamp(value)
        else:
            raise TypeError(
                "Unsupported type, expected: datetime.datetime, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @software_version.setter
    def software_version(self, value):
        if not isinstance(value, zivid.FrameInfo.SoftwareVersion):
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
        return str(zivid._frame_info_converter.to_internal_frame_info(self))
