"""Auto generated, do not edit."""
# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class CameraState:
    class Network:
        class IPV4:
            def __init__(
                self,
                address=_zivid.CameraState.Network.IPV4.Address().value,
            ):
                if isinstance(address, (str,)):
                    self._address = _zivid.CameraState.Network.IPV4.Address(address)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(
                            value_type=type(address)
                        )
                    )

            @property
            def address(self):
                return self._address.value

            @address.setter
            def address(self, value):
                if isinstance(value, (str,)):
                    self._address = _zivid.CameraState.Network.IPV4.Address(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: str, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._address == other._address:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_state_network_ipv4(self))

        def __init__(
            self,
            ipv4=None,
        ):
            if ipv4 is None:
                ipv4 = self.IPV4()
            if not isinstance(ipv4, self.IPV4):
                raise TypeError("Unsupported type: {value}".format(value=type(ipv4)))
            self._ipv4 = ipv4

        @property
        def ipv4(self):
            return self._ipv4

        @ipv4.setter
        def ipv4(self, value):
            if not isinstance(value, self.IPV4):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._ipv4 = value

        def __eq__(self, other):
            if self._ipv4 == other._ipv4:
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_state_network(self))

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

    class InaccessibleReason:
        ipConflictWithAnotherCamera = "ipConflictWithAnotherCamera"
        ipConflictWithLocalNetworkAdapter = "ipConflictWithLocalNetworkAdapter"
        ipInMultipleLocalSubnets = "ipInMultipleLocalSubnets"
        ipNotInLocalSubnet = "ipNotInLocalSubnet"

        _valid_values = {
            "ipConflictWithAnotherCamera": _zivid.CameraState.InaccessibleReason.ipConflictWithAnotherCamera,
            "ipConflictWithLocalNetworkAdapter": _zivid.CameraState.InaccessibleReason.ipConflictWithLocalNetworkAdapter,
            "ipInMultipleLocalSubnets": _zivid.CameraState.InaccessibleReason.ipInMultipleLocalSubnets,
            "ipNotInLocalSubnet": _zivid.CameraState.InaccessibleReason.ipNotInLocalSubnet,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    class Status:
        available = "available"
        busy = "busy"
        connected = "connected"
        connecting = "connecting"
        disappeared = "disappeared"
        disconnecting = "disconnecting"
        firmwareUpdateRequired = "firmwareUpdateRequired"
        inaccessible = "inaccessible"
        updatingFirmware = "updatingFirmware"

        _valid_values = {
            "available": _zivid.CameraState.Status.available,
            "busy": _zivid.CameraState.Status.busy,
            "connected": _zivid.CameraState.Status.connected,
            "connecting": _zivid.CameraState.Status.connecting,
            "disappeared": _zivid.CameraState.Status.disappeared,
            "disconnecting": _zivid.CameraState.Status.disconnecting,
            "firmwareUpdateRequired": _zivid.CameraState.Status.firmwareUpdateRequired,
            "inaccessible": _zivid.CameraState.Status.inaccessible,
            "updatingFirmware": _zivid.CameraState.Status.updatingFirmware,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    def __init__(
        self,
        available=_zivid.CameraState.Available().value,
        connected=_zivid.CameraState.Connected().value,
        inaccessible_reason=_zivid.CameraState.InaccessibleReason().value,
        status=_zivid.CameraState.Status().value,
        network=None,
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

        if (
            isinstance(inaccessible_reason, _zivid.CameraState.InaccessibleReason.enum)
            or inaccessible_reason is None
        ):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(
                inaccessible_reason
            )
        elif isinstance(inaccessible_reason, str):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(
                self.InaccessibleReason._valid_values[inaccessible_reason]
            )
        else:
            raise TypeError(
                "Unsupported type, expected: str or None, got {value_type}".format(
                    value_type=type(inaccessible_reason)
                )
            )

        if isinstance(status, _zivid.CameraState.Status.enum):
            self._status = _zivid.CameraState.Status(status)
        elif isinstance(status, str):
            self._status = _zivid.CameraState.Status(self.Status._valid_values[status])
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(status)
                )
            )

        if network is None:
            network = self.Network()
        if not isinstance(network, self.Network):
            raise TypeError("Unsupported type: {value}".format(value=type(network)))
        self._network = network

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
    def inaccessible_reason(self):
        if self._inaccessible_reason.value is None:
            return None
        for key, internal_value in self.InaccessibleReason._valid_values.items():
            if internal_value == self._inaccessible_reason.value:
                return key
        raise ValueError(
            "Unsupported value {value}".format(value=self._inaccessible_reason)
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
    def network(self):
        return self._network

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

    @inaccessible_reason.setter
    def inaccessible_reason(self, value):
        if isinstance(value, str):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(
                self.InaccessibleReason._valid_values[value]
            )
        elif (
            isinstance(value, _zivid.CameraState.InaccessibleReason.enum)
            or value is None
        ):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str or None, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @status.setter
    def status(self, value):
        if isinstance(value, str):
            self._status = _zivid.CameraState.Status(self.Status._valid_values[value])
        elif isinstance(value, _zivid.CameraState.Status.enum):
            self._status = _zivid.CameraState.Status(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @network.setter
    def network(self, value):
        if not isinstance(value, self.Network):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._network = value

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
            and self._inaccessible_reason == other._inaccessible_reason
            and self._status == other._status
            and self._network == other._network
            and self._temperature == other._temperature
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_camera_state(self))


def _to_camera_state_network_ipv4(internal_ipv4):
    return CameraState.Network.IPV4(
        address=internal_ipv4.address.value,
    )


def _to_camera_state_network(internal_network):
    return CameraState.Network(
        ipv4=_to_camera_state_network_ipv4(internal_network.ipv4),
    )


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
        network=_to_camera_state_network(internal_camera_state.network),
        temperature=_to_camera_state_temperature(internal_camera_state.temperature),
        available=internal_camera_state.available.value,
        connected=internal_camera_state.connected.value,
        inaccessible_reason=internal_camera_state.inaccessible_reason.value,
        status=internal_camera_state.status.value,
    )


def _to_internal_camera_state_network_ipv4(ipv4):
    internal_ipv4 = _zivid.CameraState.Network.IPV4()

    internal_ipv4.address = _zivid.CameraState.Network.IPV4.Address(ipv4.address)

    return internal_ipv4


def _to_internal_camera_state_network(network):
    internal_network = _zivid.CameraState.Network()

    internal_network.ipv4 = _to_internal_camera_state_network_ipv4(network.ipv4)
    return internal_network


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
    internal_camera_state.inaccessible_reason = _zivid.CameraState.InaccessibleReason(
        camera_state._inaccessible_reason.value
    )
    internal_camera_state.status = _zivid.CameraState.Status(camera_state._status.value)

    internal_camera_state.network = _to_internal_camera_state_network(
        camera_state.network
    )
    internal_camera_state.temperature = _to_internal_camera_state_temperature(
        camera_state.temperature
    )
    return internal_camera_state
