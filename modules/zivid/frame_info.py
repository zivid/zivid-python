import _zivid
import zivid


class FrameInfo:
    class SoftwareVersion:
        def __init__(
            self, core=_zivid.FrameInfo().SoftwareVersion().Core().value,
        ):

            if core is not None:
                self._core = _zivid.FrameInfo.SoftwareVersion.Core(core)
            else:
                self._core = _zivid.FrameInfo.SoftwareVersion.Core()

        @property
        def core(self):
            return self._core.value

        @core.setter
        def core(self, value):
            self._core = _zivid.FrameInfo.SoftwareVersion.Core(value)

        def __eq__(self, other):
            if self._core == other._core:
                return True
            return False

        def __str__(self):
            return """SoftwareVersion:
        core: {core}
        """.format(
                core=self.core,
            )

    def __init__(
        self, time_stamp=_zivid.FrameInfo().TimeStamp().value, software_version=None,
    ):

        if time_stamp is not None:
            self._time_stamp = _zivid.FrameInfo.TimeStamp(time_stamp)
        else:
            self._time_stamp = _zivid.FrameInfo.TimeStamp()
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
        self._time_stamp = _zivid.FrameInfo.TimeStamp(value)

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
        return """FrameInfo:
    time_stamp: {time_stamp}
    software_version: {software_version}
    """.format(
            time_stamp=self.time_stamp, software_version=self.software_version,
        )
