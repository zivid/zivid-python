import _zivid
import zivid


class CameraInfo:
    class Revision:
        def __init__(
            self,
            major=_zivid.CameraInfo().Revision().Major().value,
            minor=_zivid.CameraInfo().Revision().Minor().value,
        ):

            if major is not None:
                self._major = _zivid.CameraInfo.Revision.Major(major)
            else:
                self._major = _zivid.CameraInfo.Revision.Major()
            if minor is not None:
                self._minor = _zivid.CameraInfo.Revision.Minor(minor)
            else:
                self._minor = _zivid.CameraInfo.Revision.Minor()

        @property
        def major(self):
            return self._major.value

        @property
        def minor(self):
            return self._minor.value

        @major.setter
        def major(self, value):
            self._major = _zivid.CameraInfo.Revision.Major(value)

        @minor.setter
        def minor(self, value):
            self._minor = _zivid.CameraInfo.Revision.Minor(value)

        def __eq__(self, other):
            if self._major == other._major and self._minor == other._minor:
                return True
            return False

        def __str__(self):
            return """Revision:
        major: {major}
        minor: {minor}
        """.format(
                major=self.major, minor=self.minor,
            )

    class UserData:
        def __init__(
            self, max_size_bytes=_zivid.CameraInfo().UserData().MaxSizeBytes().value,
        ):

            if max_size_bytes is not None:
                self._max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(
                    max_size_bytes
                )
            else:
                self._max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes()

        @property
        def max_size_bytes(self):
            return self._max_size_bytes.value

        @max_size_bytes.setter
        def max_size_bytes(self, value):
            self._max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(value)

        def __eq__(self, other):
            if self._max_size_bytes == other._max_size_bytes:
                return True
            return False

        def __str__(self):
            return """UserData:
        max_size_bytes: {max_size_bytes}
        """.format(
                max_size_bytes=self.max_size_bytes,
            )

    def __init__(
        self,
        firmware_version=_zivid.CameraInfo().FirmwareVersion().value,
        model_name=_zivid.CameraInfo().ModelName().value,
        serial_number=_zivid.CameraInfo().SerialNumber().value,
        revision=None,
        user_data=None,
    ):

        if firmware_version is not None:
            self._firmware_version = _zivid.CameraInfo.FirmwareVersion(firmware_version)
        else:
            self._firmware_version = _zivid.CameraInfo.FirmwareVersion()
        if model_name is not None:
            self._model_name = _zivid.CameraInfo.ModelName(model_name)
        else:
            self._model_name = _zivid.CameraInfo.ModelName()
        if serial_number is not None:
            self._serial_number = _zivid.CameraInfo.SerialNumber(serial_number)
        else:
            self._serial_number = _zivid.CameraInfo.SerialNumber()
        if revision is None:
            revision = zivid.CameraInfo.Revision()
        if not isinstance(revision, zivid.CameraInfo.Revision):
            raise TypeError("Unsupported type: {value}".format(value=type(revision)))
        self._revision = revision
        if user_data is None:
            user_data = zivid.CameraInfo.UserData()
        if not isinstance(user_data, zivid.CameraInfo.UserData):
            raise TypeError("Unsupported type: {value}".format(value=type(user_data)))
        self._user_data = user_data

    @property
    def firmware_version(self):
        return self._firmware_version.value

    @property
    def model_name(self):
        return self._model_name.value

    @property
    def serial_number(self):
        return self._serial_number.value

    @property
    def revision(self):
        return self._revision

    @property
    def user_data(self):
        return self._user_data

    @firmware_version.setter
    def firmware_version(self, value):
        self._firmware_version = _zivid.CameraInfo.FirmwareVersion(value)

    @model_name.setter
    def model_name(self, value):
        self._model_name = _zivid.CameraInfo.ModelName(value)

    @serial_number.setter
    def serial_number(self, value):
        self._serial_number = _zivid.CameraInfo.SerialNumber(value)

    @revision.setter
    def revision(self, value):
        if not isinstance(value, zivid.CameraInfo.Revision):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._revision = value

    @user_data.setter
    def user_data(self, value):
        if not isinstance(value, zivid.CameraInfo.UserData):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._user_data = value

    def __eq__(self, other):
        if (
            self._firmware_version == other._firmware_version
            and self._model_name == other._model_name
            and self._serial_number == other._serial_number
            and self._revision == other._revision
            and self._user_data == other._user_data
        ):
            return True
        return False

    def __str__(self):
        return """CameraInfo:
    firmware_version: {firmware_version}
    model_name: {model_name}
    serial_number: {serial_number}
    revision: {revision}
    user_data: {user_data}
    """.format(
            firmware_version=self.firmware_version,
            model_name=self.model_name,
            serial_number=self.serial_number,
            revision=self.revision,
            user_data=self.user_data,
        )
