import _zivid
import zivid


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

            if dmd is not None:
                self._dmd = _zivid.CameraState.Temperature.DMD(dmd)
            else:
                self._dmd = _zivid.CameraState.Temperature.DMD()
            if general is not None:
                self._general = _zivid.CameraState.Temperature.General(general)
            else:
                self._general = _zivid.CameraState.Temperature.General()
            if led is not None:
                self._led = _zivid.CameraState.Temperature.LED(led)
            else:
                self._led = _zivid.CameraState.Temperature.LED()
            if lens is not None:
                self._lens = _zivid.CameraState.Temperature.Lens(lens)
            else:
                self._lens = _zivid.CameraState.Temperature.Lens()
            if pcb is not None:
                self._pcb = _zivid.CameraState.Temperature.PCB(pcb)
            else:
                self._pcb = _zivid.CameraState.Temperature.PCB()

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
            self._dmd = _zivid.CameraState.Temperature.DMD(value)

        @general.setter
        def general(self, value):
            self._general = _zivid.CameraState.Temperature.General(value)

        @led.setter
        def led(self, value):
            self._led = _zivid.CameraState.Temperature.LED(value)

        @lens.setter
        def lens(self, value):
            self._lens = _zivid.CameraState.Temperature.Lens(value)

        @pcb.setter
        def pcb(self, value):
            self._pcb = _zivid.CameraState.Temperature.PCB(value)

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
            return """Temperature:
        dmd: {dmd}
        general: {general}
        led: {led}
        lens: {lens}
        pcb: {pcb}
        """.format(
                dmd=self.dmd,
                general=self.general,
                led=self.led,
                lens=self.lens,
                pcb=self.pcb,
            )

    def __init__(
        self,
        available=_zivid.CameraState().Available().value,
        connected=_zivid.CameraState().Connected().value,
        temperature=None,
    ):

        if available is not None:
            self._available = _zivid.CameraState.Available(available)
        else:
            self._available = _zivid.CameraState.Available()
        if connected is not None:
            self._connected = _zivid.CameraState.Connected(connected)
        else:
            self._connected = _zivid.CameraState.Connected()
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
        self._available = _zivid.CameraState.Available(value)

    @connected.setter
    def connected(self, value):
        self._connected = _zivid.CameraState.Connected(value)

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
        return """CameraState:
    available: {available}
    connected: {connected}
    temperature: {temperature}
    """.format(
            available=self.available,
            connected=self.connected,
            temperature=self.temperature,
        )
