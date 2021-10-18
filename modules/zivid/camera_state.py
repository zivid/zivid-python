"""Auto generated, do not edit."""
# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches,too-many-boolean-expressions
import _zivid


class CameraState:
    class Temperature:
        def __init__(
            self,
            dmd=_zivid.CameraState.Temperature.DMD().value,
            general=_zivid.CameraState.Temperature.General().value,
            led=_zivid.CameraState.Temperature.LED().value,
            lens=_zivid.CameraState.Temperature.Lens().value,
            pcb=_zivid.CameraState.Temperature.PCB().value,
        ):

            if isinstance(
                dmd,
                (
                    float,
                    int,
                ),
            ):
                self._dmd = _zivid.CameraState.Temperature.DMD(dmd)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(dmd)
                    )
                )

            if isinstance(
                general,
                (
                    float,
                    int,
                ),
            ):
                self._general = _zivid.CameraState.Temperature.General(general)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(general)
                    )
                )

            if isinstance(
                led,
                (
                    float,
                    int,
                ),
            ):
                self._led = _zivid.CameraState.Temperature.LED(led)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(led)
                    )
                )

            if isinstance(
                lens,
                (
                    float,
                    int,
                ),
            ):
                self._lens = _zivid.CameraState.Temperature.Lens(lens)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(lens)
                    )
                )

            if isinstance(
                pcb,
                (
                    float,
                    int,
                ),
            ):
                self._pcb = _zivid.CameraState.Temperature.PCB(pcb)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(pcb)
                    )
                )

        @property
        def dmd(self):
            return self._dmd.value

        @property
        def general(self):
            return self._general.value

        @property
        def led(self):
            return self._led.value

        @property
        def lens(self):
            return self._lens.value

        @property
        def pcb(self):
            return self._pcb.value

        @dmd.setter
        def dmd(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._dmd = _zivid.CameraState.Temperature.DMD(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @general.setter
        def general(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._general = _zivid.CameraState.Temperature.General(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @led.setter
        def led(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._led = _zivid.CameraState.Temperature.LED(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @lens.setter
        def lens(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._lens = _zivid.CameraState.Temperature.Lens(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @pcb.setter
        def pcb(self, value):
            if isinstance(
                value,
                (
                    float,
                    int,
                ),
            ):
                self._pcb = _zivid.CameraState.Temperature.PCB(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if (
                self._dmd == other._dmd
                and self._general == other._general
                and self._led == other._led
                and self._lens == other._lens
                and self._pcb == other._pcb
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_state_temperature(self))

    def __init__(
        self,
        available=_zivid.CameraState.Available().value,
        connected=_zivid.CameraState.Connected().value,
        temperature=None,
    ):

        if isinstance(available, (bool,)):
            self._available = _zivid.CameraState.Available(available)
        else:
            raise TypeError(
                "Unsupported type, expected: (bool,), got {value_type}".format(
                    value_type=type(available)
                )
            )

        if isinstance(connected, (bool,)):
            self._connected = _zivid.CameraState.Connected(connected)
        else:
            raise TypeError(
                "Unsupported type, expected: (bool,), got {value_type}".format(
                    value_type=type(connected)
                )
            )

        if temperature is None:
            temperature = self.Temperature()
        if not isinstance(temperature, self.Temperature):
            raise TypeError("Unsupported type: {value}".format(value=type(temperature)))
        self._temperature = temperature

    @property
    def available(self):
        return self._available.value

    @property
    def connected(self):
        return self._connected.value

    @property
    def temperature(self):
        return self._temperature

    @available.setter
    def available(self, value):
        if isinstance(value, (bool,)):
            self._available = _zivid.CameraState.Available(value)
        else:
            raise TypeError(
                "Unsupported type, expected: bool, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @connected.setter
    def connected(self, value):
        if isinstance(value, (bool,)):
            self._connected = _zivid.CameraState.Connected(value)
        else:
            raise TypeError(
                "Unsupported type, expected: bool, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @temperature.setter
    def temperature(self, value):
        if not isinstance(value, self.Temperature):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._temperature = value

    @classmethod
    def load(cls, file_name):
        return _to_camera_state(_zivid.CameraState(str(file_name)))

    def save(self, file_name):
        _to_internal_camera_state(self).save(str(file_name))

    def __eq__(self, other):
        if (
            self._available == other._available
            and self._connected == other._connected
            and self._temperature == other._temperature
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_camera_state(self))


def _to_camera_state_temperature(internal_temperature):
    return CameraState.Temperature(
        dmd=internal_temperature.dmd.value,
        general=internal_temperature.general.value,
        led=internal_temperature.led.value,
        lens=internal_temperature.lens.value,
        pcb=internal_temperature.pcb.value,
    )


def _to_camera_state(internal_camera_state):
    return CameraState(
        temperature=_to_camera_state_temperature(internal_camera_state.temperature),
        available=internal_camera_state.available.value,
        connected=internal_camera_state.connected.value,
    )


def _to_internal_camera_state_temperature(temperature):
    internal_temperature = _zivid.CameraState.Temperature()

    internal_temperature.dmd = _zivid.CameraState.Temperature.DMD(temperature.dmd)
    internal_temperature.general = _zivid.CameraState.Temperature.General(
        temperature.general
    )
    internal_temperature.led = _zivid.CameraState.Temperature.LED(temperature.led)
    internal_temperature.lens = _zivid.CameraState.Temperature.Lens(temperature.lens)
    internal_temperature.pcb = _zivid.CameraState.Temperature.PCB(temperature.pcb)

    return internal_temperature


def _to_internal_camera_state(camera_state):
    internal_camera_state = _zivid.CameraState()

    internal_camera_state.available = _zivid.CameraState.Available(
        camera_state.available
    )
    internal_camera_state.connected = _zivid.CameraState.Connected(
        camera_state.connected
    )

    internal_camera_state.temperature = _to_internal_camera_state_temperature(
        camera_state.temperature
    )
    return internal_camera_state
