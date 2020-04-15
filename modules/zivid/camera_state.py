"""Auto generated, do not edit."""
# pylint: disable=too-many-arguments,missing-class-docstring,missing-function-docstring
import _zivid
import zivid
import zivid._camera_state_converter


class CameraState:
    class Temperature:
        def __init__(
            self,
            dmd=_zivid.CameraState().Temperature().DMD().value,
            general=_zivid.CameraState().Temperature().General().value,
            led=_zivid.CameraState().Temperature().LED().value,
            lens=_zivid.CameraState().Temperature().Lens().value,
            pcb=_zivid.CameraState().Temperature().PCB().value,
        ):

            if isinstance(dmd, (float, int,)):
                self._dmd = _zivid.CameraState.Temperature.DMD(dmd)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(dmd)
                    )
                )
            if isinstance(general, (float, int,)):
                self._general = _zivid.CameraState.Temperature.General(general)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(general)
                    )
                )
            if isinstance(led, (float, int,)):
                self._led = _zivid.CameraState.Temperature.LED(led)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(led)
                    )
                )
            if isinstance(lens, (float, int,)):
                self._lens = _zivid.CameraState.Temperature.Lens(lens)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,), got {value_type}".format(
                        value_type=type(lens)
                    )
                )
            if isinstance(pcb, (float, int,)):
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
            if isinstance(value, (float, int,)):
                self._dmd = _zivid.CameraState.Temperature.DMD(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @general.setter
        def general(self, value):
            if isinstance(value, (float, int,)):
                self._general = _zivid.CameraState.Temperature.General(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @led.setter
        def led(self, value):
            if isinstance(value, (float, int,)):
                self._led = _zivid.CameraState.Temperature.LED(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @lens.setter
        def lens(self, value):
            if isinstance(value, (float, int,)):
                self._lens = _zivid.CameraState.Temperature.Lens(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @pcb.setter
        def pcb(self, value):
            if isinstance(value, (float, int,)):
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
            return str(
                zivid._camera_state_converter.to_internal_camera_state_temperature(self)
            )

    def __init__(
        self,
        available=_zivid.CameraState().Available().value,
        connected=_zivid.CameraState().Connected().value,
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
            temperature = zivid.CameraState.Temperature()
        if not isinstance(temperature, zivid.CameraState.Temperature):
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
        if not isinstance(value, zivid.CameraState.Temperature):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._temperature = value

    def __eq__(self, other):
        if (
            self._available == other._available
            and self._connected == other._connected
            and self._temperature == other._temperature
        ):
            return True
        return False

    def __str__(self):
        return str(zivid._camera_state_converter.to_internal_camera_state(self))
