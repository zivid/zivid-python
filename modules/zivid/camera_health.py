"""Auto generated, do not edit."""

from __future__ import annotations

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime

import _zivid


class CameraHealth:

    class Fan:

        class Status:

            Error = "Error"
            OK = "OK"
            Suboptimal = "Suboptimal"
            Unknown = "Unknown"

            _valid_values = {
                "Error": _zivid.CameraHealth.Fan.Status.Error,
                "OK": _zivid.CameraHealth.Fan.Status.OK,
                "Suboptimal": _zivid.CameraHealth.Fan.Status.Suboptimal,
                "Unknown": _zivid.CameraHealth.Fan.Status.Unknown,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        class Value:

            ok = "ok"
            unexpectedFanStop = "unexpectedFanStop"

            _valid_values = {
                "ok": _zivid.CameraHealth.Fan.Value.ok,
                "unexpectedFanStop": _zivid.CameraHealth.Fan.Value.unexpectedFanStop,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            status: str = _zivid.CameraHealth.Fan.Status().value,
            value: str | None = _zivid.CameraHealth.Fan.Value().value,
        ):

            if isinstance(status, _zivid.CameraHealth.Fan.Status.enum):
                self._status = _zivid.CameraHealth.Fan.Status(status)
            elif isinstance(status, str):
                self._status = _zivid.CameraHealth.Fan.Status(self.Status._valid_values[status])
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

            if isinstance(value, _zivid.CameraHealth.Fan.Value.enum) or value is None:
                self._value = _zivid.CameraHealth.Fan.Value(value)
            elif isinstance(value, str):
                self._value = _zivid.CameraHealth.Fan.Value(self.Value._valid_values[value])
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        @property
        def status(self):
            if self._status.value is None:
                return None
            for key, internal_value in self.Status._valid_values.items():
                if internal_value == self._status.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._status))

        @property
        def value(self):
            if self._value.value is None:
                return None
            for key, internal_value in self.Value._valid_values.items():
                if internal_value == self._value.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._value))

        @status.setter
        def status(self, value):
            if isinstance(value, str):
                self._status = _zivid.CameraHealth.Fan.Status(self.Status._valid_values[value])
            elif isinstance(value, _zivid.CameraHealth.Fan.Status.enum):
                self._status = _zivid.CameraHealth.Fan.Status(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

        @value.setter
        def value(self, value):
            if isinstance(value, str):
                self._value = _zivid.CameraHealth.Fan.Value(self.Value._valid_values[value])
            elif isinstance(value, _zivid.CameraHealth.Fan.Value.enum) or value is None:
                self._value = _zivid.CameraHealth.Fan.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._status == other._status and self._value == other._value:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_health_fan(self))

    class InfieldVerification:

        class Status:

            Error = "Error"
            OK = "OK"
            Suboptimal = "Suboptimal"
            Unknown = "Unknown"

            _valid_values = {
                "Error": _zivid.CameraHealth.InfieldVerification.Status.Error,
                "OK": _zivid.CameraHealth.InfieldVerification.Status.OK,
                "Suboptimal": _zivid.CameraHealth.InfieldVerification.Status.Suboptimal,
                "Unknown": _zivid.CameraHealth.InfieldVerification.Status.Unknown,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            status: str = _zivid.CameraHealth.InfieldVerification.Status().value,
            value: datetime.datetime | None = _zivid.CameraHealth.InfieldVerification.Value().value,
        ):

            if isinstance(status, _zivid.CameraHealth.InfieldVerification.Status.enum):
                self._status = _zivid.CameraHealth.InfieldVerification.Status(status)
            elif isinstance(status, str):
                self._status = _zivid.CameraHealth.InfieldVerification.Status(self.Status._valid_values[status])
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

            if isinstance(value, (datetime.datetime,)) or value is None:
                self._value = _zivid.CameraHealth.InfieldVerification.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.datetime,) or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @property
        def status(self):
            if self._status.value is None:
                return None
            for key, internal_value in self.Status._valid_values.items():
                if internal_value == self._status.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._status))

        @property
        def value(self):
            return self._value.value

        @status.setter
        def status(self, value):
            if isinstance(value, str):
                self._status = _zivid.CameraHealth.InfieldVerification.Status(self.Status._valid_values[value])
            elif isinstance(value, _zivid.CameraHealth.InfieldVerification.Status.enum):
                self._status = _zivid.CameraHealth.InfieldVerification.Status(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

        @value.setter
        def value(self, value):
            if isinstance(value, (datetime.datetime,)) or value is None:
                self._value = _zivid.CameraHealth.InfieldVerification.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.datetime or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._status == other._status and self._value == other._value:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_health_infield_verification(self))

    class MaxTransferSpeed:

        class Status:

            Error = "Error"
            OK = "OK"
            Suboptimal = "Suboptimal"
            Unknown = "Unknown"

            _valid_values = {
                "Error": _zivid.CameraHealth.MaxTransferSpeed.Status.Error,
                "OK": _zivid.CameraHealth.MaxTransferSpeed.Status.OK,
                "Suboptimal": _zivid.CameraHealth.MaxTransferSpeed.Status.Suboptimal,
                "Unknown": _zivid.CameraHealth.MaxTransferSpeed.Status.Unknown,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            status: str = _zivid.CameraHealth.MaxTransferSpeed.Status().value,
            value: int | None = _zivid.CameraHealth.MaxTransferSpeed.Value().value,
        ):

            if isinstance(status, _zivid.CameraHealth.MaxTransferSpeed.Status.enum):
                self._status = _zivid.CameraHealth.MaxTransferSpeed.Status(status)
            elif isinstance(status, str):
                self._status = _zivid.CameraHealth.MaxTransferSpeed.Status(self.Status._valid_values[status])
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

            if isinstance(value, (int,)) or value is None:
                self._value = _zivid.CameraHealth.MaxTransferSpeed.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: (int,) or None, got {value_type}".format(value_type=type(value))
                )

        @property
        def status(self):
            if self._status.value is None:
                return None
            for key, internal_value in self.Status._valid_values.items():
                if internal_value == self._status.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._status))

        @property
        def value(self):
            return self._value.value

        @status.setter
        def status(self, value):
            if isinstance(value, str):
                self._status = _zivid.CameraHealth.MaxTransferSpeed.Status(self.Status._valid_values[value])
            elif isinstance(value, _zivid.CameraHealth.MaxTransferSpeed.Status.enum):
                self._status = _zivid.CameraHealth.MaxTransferSpeed.Status(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

        @value.setter
        def value(self, value):
            if isinstance(value, (int,)) or value is None:
                self._value = _zivid.CameraHealth.MaxTransferSpeed.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: int or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._status == other._status and self._value == other._value:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_health_max_transfer_speed(self))

    class Memory:

        class Status:

            Error = "Error"
            OK = "OK"
            Suboptimal = "Suboptimal"
            Unknown = "Unknown"

            _valid_values = {
                "Error": _zivid.CameraHealth.Memory.Status.Error,
                "OK": _zivid.CameraHealth.Memory.Status.OK,
                "Suboptimal": _zivid.CameraHealth.Memory.Status.Suboptimal,
                "Unknown": _zivid.CameraHealth.Memory.Status.Unknown,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            status: str = _zivid.CameraHealth.Memory.Status().value,
            value: int | None = _zivid.CameraHealth.Memory.Value().value,
        ):

            if isinstance(status, _zivid.CameraHealth.Memory.Status.enum):
                self._status = _zivid.CameraHealth.Memory.Status(status)
            elif isinstance(status, str):
                self._status = _zivid.CameraHealth.Memory.Status(self.Status._valid_values[status])
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

            if isinstance(value, (int,)) or value is None:
                self._value = _zivid.CameraHealth.Memory.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: (int,) or None, got {value_type}".format(value_type=type(value))
                )

        @property
        def status(self):
            if self._status.value is None:
                return None
            for key, internal_value in self.Status._valid_values.items():
                if internal_value == self._status.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._status))

        @property
        def value(self):
            return self._value.value

        @status.setter
        def status(self, value):
            if isinstance(value, str):
                self._status = _zivid.CameraHealth.Memory.Status(self.Status._valid_values[value])
            elif isinstance(value, _zivid.CameraHealth.Memory.Status.enum):
                self._status = _zivid.CameraHealth.Memory.Status(value)
            else:
                raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

        @value.setter
        def value(self, value):
            if isinstance(value, (int,)) or value is None:
                self._value = _zivid.CameraHealth.Memory.Value(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: int or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._status == other._status and self._value == other._value:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_health_memory(self))

    class Temperature:

        class DMD:

            class Status:

                Error = "Error"
                OK = "OK"
                Suboptimal = "Suboptimal"
                Unknown = "Unknown"

                _valid_values = {
                    "Error": _zivid.CameraHealth.Temperature.DMD.Status.Error,
                    "OK": _zivid.CameraHealth.Temperature.DMD.Status.OK,
                    "Suboptimal": _zivid.CameraHealth.Temperature.DMD.Status.Suboptimal,
                    "Unknown": _zivid.CameraHealth.Temperature.DMD.Status.Unknown,
                }

                @classmethod
                def valid_values(cls):
                    return list(cls._valid_values.keys())

            def __init__(
                self,
                status: str = _zivid.CameraHealth.Temperature.DMD.Status().value,
                value: float | int | None = _zivid.CameraHealth.Temperature.DMD.Value().value,
            ):

                if isinstance(status, _zivid.CameraHealth.Temperature.DMD.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.DMD.Status(status)
                elif isinstance(status, str):
                    self._status = _zivid.CameraHealth.Temperature.DMD.Status(self.Status._valid_values[status])
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.DMD.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @property
            def status(self):
                if self._status.value is None:
                    return None
                for key, internal_value in self.Status._valid_values.items():
                    if internal_value == self._status.value:
                        return key
                raise ValueError("Unsupported value {value}".format(value=self._status))

            @property
            def value(self):
                return self._value.value

            @status.setter
            def status(self, value):
                if isinstance(value, str):
                    self._status = _zivid.CameraHealth.Temperature.DMD.Status(self.Status._valid_values[value])
                elif isinstance(value, _zivid.CameraHealth.Temperature.DMD.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.DMD.Status(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            @value.setter
            def value(self, value):
                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.DMD.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._status == other._status and self._value == other._value:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_health_temperature_dmd(self))

        class LED:

            class Status:

                Error = "Error"
                OK = "OK"
                Suboptimal = "Suboptimal"
                Unknown = "Unknown"

                _valid_values = {
                    "Error": _zivid.CameraHealth.Temperature.LED.Status.Error,
                    "OK": _zivid.CameraHealth.Temperature.LED.Status.OK,
                    "Suboptimal": _zivid.CameraHealth.Temperature.LED.Status.Suboptimal,
                    "Unknown": _zivid.CameraHealth.Temperature.LED.Status.Unknown,
                }

                @classmethod
                def valid_values(cls):
                    return list(cls._valid_values.keys())

            def __init__(
                self,
                status: str = _zivid.CameraHealth.Temperature.LED.Status().value,
                value: float | int | None = _zivid.CameraHealth.Temperature.LED.Value().value,
            ):

                if isinstance(status, _zivid.CameraHealth.Temperature.LED.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.LED.Status(status)
                elif isinstance(status, str):
                    self._status = _zivid.CameraHealth.Temperature.LED.Status(self.Status._valid_values[status])
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.LED.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @property
            def status(self):
                if self._status.value is None:
                    return None
                for key, internal_value in self.Status._valid_values.items():
                    if internal_value == self._status.value:
                        return key
                raise ValueError("Unsupported value {value}".format(value=self._status))

            @property
            def value(self):
                return self._value.value

            @status.setter
            def status(self, value):
                if isinstance(value, str):
                    self._status = _zivid.CameraHealth.Temperature.LED.Status(self.Status._valid_values[value])
                elif isinstance(value, _zivid.CameraHealth.Temperature.LED.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.LED.Status(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            @value.setter
            def value(self, value):
                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.LED.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._status == other._status and self._value == other._value:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_health_temperature_led(self))

        class Lens:

            class Status:

                Error = "Error"
                OK = "OK"
                Suboptimal = "Suboptimal"
                Unknown = "Unknown"

                _valid_values = {
                    "Error": _zivid.CameraHealth.Temperature.Lens.Status.Error,
                    "OK": _zivid.CameraHealth.Temperature.Lens.Status.OK,
                    "Suboptimal": _zivid.CameraHealth.Temperature.Lens.Status.Suboptimal,
                    "Unknown": _zivid.CameraHealth.Temperature.Lens.Status.Unknown,
                }

                @classmethod
                def valid_values(cls):
                    return list(cls._valid_values.keys())

            def __init__(
                self,
                status: str = _zivid.CameraHealth.Temperature.Lens.Status().value,
                value: float | int | None = _zivid.CameraHealth.Temperature.Lens.Value().value,
            ):

                if isinstance(status, _zivid.CameraHealth.Temperature.Lens.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.Lens.Status(status)
                elif isinstance(status, str):
                    self._status = _zivid.CameraHealth.Temperature.Lens.Status(self.Status._valid_values[status])
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.Lens.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @property
            def status(self):
                if self._status.value is None:
                    return None
                for key, internal_value in self.Status._valid_values.items():
                    if internal_value == self._status.value:
                        return key
                raise ValueError("Unsupported value {value}".format(value=self._status))

            @property
            def value(self):
                return self._value.value

            @status.setter
            def status(self, value):
                if isinstance(value, str):
                    self._status = _zivid.CameraHealth.Temperature.Lens.Status(self.Status._valid_values[value])
                elif isinstance(value, _zivid.CameraHealth.Temperature.Lens.Status.enum):
                    self._status = _zivid.CameraHealth.Temperature.Lens.Status(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            @value.setter
            def value(self, value):
                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._value = _zivid.CameraHealth.Temperature.Lens.Value(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._status == other._status and self._value == other._value:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_health_temperature_lens(self))

        def __init__(
            self,
            dmd: CameraHealth.Temperature.DMD | None = None,
            led: CameraHealth.Temperature.LED | None = None,
            lens: CameraHealth.Temperature.Lens | None = None,
        ):

            if dmd is None:
                dmd = self.DMD()
            if not isinstance(dmd, self.DMD):
                raise TypeError("Unsupported type: {value}".format(value=type(dmd)))
            self._dmd = dmd

            if led is None:
                led = self.LED()
            if not isinstance(led, self.LED):
                raise TypeError("Unsupported type: {value}".format(value=type(led)))
            self._led = led

            if lens is None:
                lens = self.Lens()
            if not isinstance(lens, self.Lens):
                raise TypeError("Unsupported type: {value}".format(value=type(lens)))
            self._lens = lens

        @property
        def dmd(self):
            return self._dmd

        @property
        def led(self):
            return self._led

        @property
        def lens(self):
            return self._lens

        @dmd.setter
        def dmd(self, value):
            if not isinstance(value, self.DMD):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._dmd = value

        @led.setter
        def led(self, value):
            if not isinstance(value, self.LED):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._led = value

        @lens.setter
        def lens(self, value):
            if not isinstance(value, self.Lens):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._lens = value

        def __eq__(self, other):
            if self._dmd == other._dmd and self._led == other._led and self._lens == other._lens:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_health_temperature(self))

    class Overall:

        Error = "Error"
        OK = "OK"
        Suboptimal = "Suboptimal"
        Unknown = "Unknown"

        _valid_values = {
            "Error": _zivid.CameraHealth.Overall.Error,
            "OK": _zivid.CameraHealth.Overall.OK,
            "Suboptimal": _zivid.CameraHealth.Overall.Suboptimal,
            "Unknown": _zivid.CameraHealth.Overall.Unknown,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    def __init__(
        self,
        overall: str = _zivid.CameraHealth.Overall().value,
        fan: CameraHealth.Fan | None = None,
        infield_verification: CameraHealth.InfieldVerification | None = None,
        max_transfer_speed: CameraHealth.MaxTransferSpeed | None = None,
        memory: CameraHealth.Memory | None = None,
        temperature: CameraHealth.Temperature | None = None,
    ):

        if isinstance(overall, _zivid.CameraHealth.Overall.enum):
            self._overall = _zivid.CameraHealth.Overall(overall)
        elif isinstance(overall, str):
            self._overall = _zivid.CameraHealth.Overall(self.Overall._valid_values[overall])
        else:
            raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(overall)))

        if fan is None:
            fan = self.Fan()
        if not isinstance(fan, self.Fan):
            raise TypeError("Unsupported type: {value}".format(value=type(fan)))
        self._fan = fan

        if infield_verification is None:
            infield_verification = self.InfieldVerification()
        if not isinstance(infield_verification, self.InfieldVerification):
            raise TypeError("Unsupported type: {value}".format(value=type(infield_verification)))
        self._infield_verification = infield_verification

        if max_transfer_speed is None:
            max_transfer_speed = self.MaxTransferSpeed()
        if not isinstance(max_transfer_speed, self.MaxTransferSpeed):
            raise TypeError("Unsupported type: {value}".format(value=type(max_transfer_speed)))
        self._max_transfer_speed = max_transfer_speed

        if memory is None:
            memory = self.Memory()
        if not isinstance(memory, self.Memory):
            raise TypeError("Unsupported type: {value}".format(value=type(memory)))
        self._memory = memory

        if temperature is None:
            temperature = self.Temperature()
        if not isinstance(temperature, self.Temperature):
            raise TypeError("Unsupported type: {value}".format(value=type(temperature)))
        self._temperature = temperature

    @property
    def overall(self):
        if self._overall.value is None:
            return None
        for key, internal_value in self.Overall._valid_values.items():
            if internal_value == self._overall.value:
                return key
        raise ValueError("Unsupported value {value}".format(value=self._overall))

    @property
    def fan(self):
        return self._fan

    @property
    def infield_verification(self):
        return self._infield_verification

    @property
    def max_transfer_speed(self):
        return self._max_transfer_speed

    @property
    def memory(self):
        return self._memory

    @property
    def temperature(self):
        return self._temperature

    @overall.setter
    def overall(self, value):
        if isinstance(value, str):
            self._overall = _zivid.CameraHealth.Overall(self.Overall._valid_values[value])
        elif isinstance(value, _zivid.CameraHealth.Overall.enum):
            self._overall = _zivid.CameraHealth.Overall(value)
        else:
            raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

    @fan.setter
    def fan(self, value):
        if not isinstance(value, self.Fan):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._fan = value

    @infield_verification.setter
    def infield_verification(self, value):
        if not isinstance(value, self.InfieldVerification):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._infield_verification = value

    @max_transfer_speed.setter
    def max_transfer_speed(self, value):
        if not isinstance(value, self.MaxTransferSpeed):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._max_transfer_speed = value

    @memory.setter
    def memory(self, value):
        if not isinstance(value, self.Memory):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._memory = value

    @temperature.setter
    def temperature(self, value):
        if not isinstance(value, self.Temperature):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._temperature = value

    @classmethod
    def load(cls, file_name):
        return _to_camera_health(_zivid.CameraHealth(str(file_name)))

    def save(self, file_name):
        _to_internal_camera_health(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_camera_health(_zivid.CameraHealth.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_camera_health(self).serialize()

    def __eq__(self, other):
        if (
            self._overall == other._overall
            and self._fan == other._fan
            and self._infield_verification == other._infield_verification
            and self._max_transfer_speed == other._max_transfer_speed
            and self._memory == other._memory
            and self._temperature == other._temperature
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_camera_health(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_camera_health(_to_internal_camera_health(self))


def _to_camera_health_fan(internal_fan):
    return CameraHealth.Fan(
        status=internal_fan.status.value,
        value=internal_fan.value.value,
    )


def _to_camera_health_infield_verification(internal_infield_verification):
    return CameraHealth.InfieldVerification(
        status=internal_infield_verification.status.value,
        value=internal_infield_verification.value.value,
    )


def _to_camera_health_max_transfer_speed(internal_max_transfer_speed):
    return CameraHealth.MaxTransferSpeed(
        status=internal_max_transfer_speed.status.value,
        value=internal_max_transfer_speed.value.value,
    )


def _to_camera_health_memory(internal_memory):
    return CameraHealth.Memory(
        status=internal_memory.status.value,
        value=internal_memory.value.value,
    )


def _to_camera_health_temperature_dmd(internal_dmd):
    return CameraHealth.Temperature.DMD(
        status=internal_dmd.status.value,
        value=internal_dmd.value.value,
    )


def _to_camera_health_temperature_led(internal_led):
    return CameraHealth.Temperature.LED(
        status=internal_led.status.value,
        value=internal_led.value.value,
    )


def _to_camera_health_temperature_lens(internal_lens):
    return CameraHealth.Temperature.Lens(
        status=internal_lens.status.value,
        value=internal_lens.value.value,
    )


def _to_camera_health_temperature(internal_temperature):
    return CameraHealth.Temperature(
        dmd=_to_camera_health_temperature_dmd(internal_temperature.dmd),
        led=_to_camera_health_temperature_led(internal_temperature.led),
        lens=_to_camera_health_temperature_lens(internal_temperature.lens),
    )


def _to_camera_health(internal_camera_health):
    return CameraHealth(
        fan=_to_camera_health_fan(internal_camera_health.fan),
        infield_verification=_to_camera_health_infield_verification(internal_camera_health.infield_verification),
        max_transfer_speed=_to_camera_health_max_transfer_speed(internal_camera_health.max_transfer_speed),
        memory=_to_camera_health_memory(internal_camera_health.memory),
        temperature=_to_camera_health_temperature(internal_camera_health.temperature),
        overall=internal_camera_health.overall.value,
    )


def _to_internal_camera_health_fan(fan):
    internal_fan = _zivid.CameraHealth.Fan()

    internal_fan.status = _zivid.CameraHealth.Fan.Status(fan._status.value)
    internal_fan.value = _zivid.CameraHealth.Fan.Value(fan._value.value)

    return internal_fan


def _to_internal_camera_health_infield_verification(infield_verification):
    internal_infield_verification = _zivid.CameraHealth.InfieldVerification()

    internal_infield_verification.status = _zivid.CameraHealth.InfieldVerification.Status(
        infield_verification._status.value
    )
    internal_infield_verification.value = _zivid.CameraHealth.InfieldVerification.Value(infield_verification.value)

    return internal_infield_verification


def _to_internal_camera_health_max_transfer_speed(max_transfer_speed):
    internal_max_transfer_speed = _zivid.CameraHealth.MaxTransferSpeed()

    internal_max_transfer_speed.status = _zivid.CameraHealth.MaxTransferSpeed.Status(max_transfer_speed._status.value)
    internal_max_transfer_speed.value = _zivid.CameraHealth.MaxTransferSpeed.Value(max_transfer_speed.value)

    return internal_max_transfer_speed


def _to_internal_camera_health_memory(memory):
    internal_memory = _zivid.CameraHealth.Memory()

    internal_memory.status = _zivid.CameraHealth.Memory.Status(memory._status.value)
    internal_memory.value = _zivid.CameraHealth.Memory.Value(memory.value)

    return internal_memory


def _to_internal_camera_health_temperature_dmd(dmd):
    internal_dmd = _zivid.CameraHealth.Temperature.DMD()

    internal_dmd.status = _zivid.CameraHealth.Temperature.DMD.Status(dmd._status.value)
    internal_dmd.value = _zivid.CameraHealth.Temperature.DMD.Value(dmd.value)

    return internal_dmd


def _to_internal_camera_health_temperature_led(led):
    internal_led = _zivid.CameraHealth.Temperature.LED()

    internal_led.status = _zivid.CameraHealth.Temperature.LED.Status(led._status.value)
    internal_led.value = _zivid.CameraHealth.Temperature.LED.Value(led.value)

    return internal_led


def _to_internal_camera_health_temperature_lens(lens):
    internal_lens = _zivid.CameraHealth.Temperature.Lens()

    internal_lens.status = _zivid.CameraHealth.Temperature.Lens.Status(lens._status.value)
    internal_lens.value = _zivid.CameraHealth.Temperature.Lens.Value(lens.value)

    return internal_lens


def _to_internal_camera_health_temperature(temperature):
    internal_temperature = _zivid.CameraHealth.Temperature()

    internal_temperature.dmd = _to_internal_camera_health_temperature_dmd(temperature.dmd)
    internal_temperature.led = _to_internal_camera_health_temperature_led(temperature.led)
    internal_temperature.lens = _to_internal_camera_health_temperature_lens(temperature.lens)
    return internal_temperature


def _to_internal_camera_health(camera_health):
    internal_camera_health = _zivid.CameraHealth()

    internal_camera_health.overall = _zivid.CameraHealth.Overall(camera_health._overall.value)

    internal_camera_health.fan = _to_internal_camera_health_fan(camera_health.fan)
    internal_camera_health.infield_verification = _to_internal_camera_health_infield_verification(
        camera_health.infield_verification
    )
    internal_camera_health.max_transfer_speed = _to_internal_camera_health_max_transfer_speed(
        camera_health.max_transfer_speed
    )
    internal_camera_health.memory = _to_internal_camera_health_memory(camera_health.memory)
    internal_camera_health.temperature = _to_internal_camera_health_temperature(camera_health.temperature)
    return internal_camera_health
