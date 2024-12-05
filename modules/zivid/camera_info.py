"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class CameraInfo:

    class Revision:

        def __init__(
            self,
            major=_zivid.CameraInfo.Revision.Major().value,
            minor=_zivid.CameraInfo.Revision.Minor().value,
        ):

            if isinstance(major, (int,)):
                self._major = _zivid.CameraInfo.Revision.Major(major)
            else:
                raise TypeError(
                    "Unsupported type, expected: (int,), got {value_type}".format(
                        value_type=type(major)
                    )
                )

            if isinstance(minor, (int,)):
                self._minor = _zivid.CameraInfo.Revision.Minor(minor)
            else:
                raise TypeError(
                    "Unsupported type, expected: (int,), got {value_type}".format(
                        value_type=type(minor)
                    )
                )

        @property
        def major(self):
            return self._major.value

        @property
        def minor(self):
            return self._minor.value

        @major.setter
        def major(self, value):
            if isinstance(value, (int,)):
                self._major = _zivid.CameraInfo.Revision.Major(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @minor.setter
        def minor(self, value):
            if isinstance(value, (int,)):
                self._minor = _zivid.CameraInfo.Revision.Minor(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._major == other._major and self._minor == other._minor:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_info_revision(self))

    class UserData:

        def __init__(
            self,
            max_size_bytes=_zivid.CameraInfo.UserData.MaxSizeBytes().value,
        ):

            if isinstance(max_size_bytes, (int,)):
                self._max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(
                    max_size_bytes
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (int,), got {value_type}".format(
                        value_type=type(max_size_bytes)
                    )
                )

        @property
        def max_size_bytes(self):
            return self._max_size_bytes.value

        @max_size_bytes.setter
        def max_size_bytes(self, value):
            if isinstance(value, (int,)):
                self._max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._max_size_bytes == other._max_size_bytes:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_info_user_data(self))

    class Model:

        zivid2PlusL110 = "zivid2PlusL110"
        zivid2PlusLR110 = "zivid2PlusLR110"
        zivid2PlusM130 = "zivid2PlusM130"
        zivid2PlusM60 = "zivid2PlusM60"
        zivid2PlusMR130 = "zivid2PlusMR130"
        zivid2PlusMR60 = "zivid2PlusMR60"
        zividOnePlusLarge = "zividOnePlusLarge"
        zividOnePlusMedium = "zividOnePlusMedium"
        zividOnePlusSmall = "zividOnePlusSmall"
        zividTwo = "zividTwo"
        zividTwoL100 = "zividTwoL100"

        _valid_values = {
            "zivid2PlusL110": _zivid.CameraInfo.Model.zivid2PlusL110,
            "zivid2PlusLR110": _zivid.CameraInfo.Model.zivid2PlusLR110,
            "zivid2PlusM130": _zivid.CameraInfo.Model.zivid2PlusM130,
            "zivid2PlusM60": _zivid.CameraInfo.Model.zivid2PlusM60,
            "zivid2PlusMR130": _zivid.CameraInfo.Model.zivid2PlusMR130,
            "zivid2PlusMR60": _zivid.CameraInfo.Model.zivid2PlusMR60,
            "zividOnePlusLarge": _zivid.CameraInfo.Model.zividOnePlusLarge,
            "zividOnePlusMedium": _zivid.CameraInfo.Model.zividOnePlusMedium,
            "zividOnePlusSmall": _zivid.CameraInfo.Model.zividOnePlusSmall,
            "zividTwo": _zivid.CameraInfo.Model.zividTwo,
            "zividTwoL100": _zivid.CameraInfo.Model.zividTwoL100,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    def __init__(
        self,
        firmware_version=_zivid.CameraInfo.FirmwareVersion().value,
        hardware_revision=_zivid.CameraInfo.HardwareRevision().value,
        model=_zivid.CameraInfo.Model().value,
        model_name=_zivid.CameraInfo.ModelName().value,
        serial_number=_zivid.CameraInfo.SerialNumber().value,
        revision=None,
        user_data=None,
    ):

        if isinstance(firmware_version, (str,)):
            self._firmware_version = _zivid.CameraInfo.FirmwareVersion(firmware_version)
        else:
            raise TypeError(
                "Unsupported type, expected: (str,), got {value_type}".format(
                    value_type=type(firmware_version)
                )
            )

        if isinstance(hardware_revision, (str,)):
            self._hardware_revision = _zivid.CameraInfo.HardwareRevision(
                hardware_revision
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (str,), got {value_type}".format(
                    value_type=type(hardware_revision)
                )
            )

        if isinstance(model, _zivid.CameraInfo.Model.enum):
            self._model = _zivid.CameraInfo.Model(model)
        elif isinstance(model, str):
            self._model = _zivid.CameraInfo.Model(self.Model._valid_values[model])
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(model)
                )
            )

        if isinstance(model_name, (str,)):
            self._model_name = _zivid.CameraInfo.ModelName(model_name)
        else:
            raise TypeError(
                "Unsupported type, expected: (str,), got {value_type}".format(
                    value_type=type(model_name)
                )
            )

        if isinstance(serial_number, (str,)):
            self._serial_number = _zivid.CameraInfo.SerialNumber(serial_number)
        else:
            raise TypeError(
                "Unsupported type, expected: (str,), got {value_type}".format(
                    value_type=type(serial_number)
                )
            )

        if revision is None:
            revision = self.Revision()
        if not isinstance(revision, self.Revision):
            raise TypeError("Unsupported type: {value}".format(value=type(revision)))
        self._revision = revision

        if user_data is None:
            user_data = self.UserData()
        if not isinstance(user_data, self.UserData):
            raise TypeError("Unsupported type: {value}".format(value=type(user_data)))
        self._user_data = user_data

    @property
    def firmware_version(self):
        return self._firmware_version.value

    @property
    def hardware_revision(self):
        return self._hardware_revision.value

    @property
    def model(self):
        if self._model.value is None:
            return None
        for key, internal_value in self.Model._valid_values.items():
            if internal_value == self._model.value:
                return key
        raise ValueError("Unsupported value {value}".format(value=self._model))

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
        if isinstance(value, (str,)):
            self._firmware_version = _zivid.CameraInfo.FirmwareVersion(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @hardware_revision.setter
    def hardware_revision(self, value):
        if isinstance(value, (str,)):
            self._hardware_revision = _zivid.CameraInfo.HardwareRevision(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @model.setter
    def model(self, value):
        if isinstance(value, str):
            self._model = _zivid.CameraInfo.Model(self.Model._valid_values[value])
        elif isinstance(value, _zivid.CameraInfo.Model.enum):
            self._model = _zivid.CameraInfo.Model(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @model_name.setter
    def model_name(self, value):
        if isinstance(value, (str,)):
            self._model_name = _zivid.CameraInfo.ModelName(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @serial_number.setter
    def serial_number(self, value):
        if isinstance(value, (str,)):
            self._serial_number = _zivid.CameraInfo.SerialNumber(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @revision.setter
    def revision(self, value):
        if not isinstance(value, self.Revision):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._revision = value

    @user_data.setter
    def user_data(self, value):
        if not isinstance(value, self.UserData):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._user_data = value

    @classmethod
    def load(cls, file_name):
        return _to_camera_info(_zivid.CameraInfo(str(file_name)))

    def save(self, file_name):
        _to_internal_camera_info(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_camera_info(_zivid.CameraInfo.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_camera_info(self).serialize()

    def __eq__(self, other):
        if (
            self._firmware_version == other._firmware_version
            and self._hardware_revision == other._hardware_revision
            and self._model == other._model
            and self._model_name == other._model_name
            and self._serial_number == other._serial_number
            and self._revision == other._revision
            and self._user_data == other._user_data
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_camera_info(self))


def _to_camera_info_revision(internal_revision):
    return CameraInfo.Revision(
        major=internal_revision.major.value,
        minor=internal_revision.minor.value,
    )


def _to_camera_info_user_data(internal_user_data):
    return CameraInfo.UserData(
        max_size_bytes=internal_user_data.max_size_bytes.value,
    )


def _to_camera_info(internal_camera_info):
    return CameraInfo(
        revision=_to_camera_info_revision(internal_camera_info.revision),
        user_data=_to_camera_info_user_data(internal_camera_info.user_data),
        firmware_version=internal_camera_info.firmware_version.value,
        hardware_revision=internal_camera_info.hardware_revision.value,
        model=internal_camera_info.model.value,
        model_name=internal_camera_info.model_name.value,
        serial_number=internal_camera_info.serial_number.value,
    )


def _to_internal_camera_info_revision(revision):
    internal_revision = _zivid.CameraInfo.Revision()

    internal_revision.major = _zivid.CameraInfo.Revision.Major(revision.major)
    internal_revision.minor = _zivid.CameraInfo.Revision.Minor(revision.minor)

    return internal_revision


def _to_internal_camera_info_user_data(user_data):
    internal_user_data = _zivid.CameraInfo.UserData()

    internal_user_data.max_size_bytes = _zivid.CameraInfo.UserData.MaxSizeBytes(
        user_data.max_size_bytes
    )

    return internal_user_data


def _to_internal_camera_info(camera_info):
    internal_camera_info = _zivid.CameraInfo()

    internal_camera_info.firmware_version = _zivid.CameraInfo.FirmwareVersion(
        camera_info.firmware_version
    )
    internal_camera_info.hardware_revision = _zivid.CameraInfo.HardwareRevision(
        camera_info.hardware_revision
    )
    internal_camera_info.model = _zivid.CameraInfo.Model(camera_info._model.value)
    internal_camera_info.model_name = _zivid.CameraInfo.ModelName(
        camera_info.model_name
    )
    internal_camera_info.serial_number = _zivid.CameraInfo.SerialNumber(
        camera_info.serial_number
    )

    internal_camera_info.revision = _to_internal_camera_info_revision(
        camera_info.revision
    )
    internal_camera_info.user_data = _to_internal_camera_info_user_data(
        camera_info.user_data
    )
    return internal_camera_info
