"""Auto generated, do not edit."""

from __future__ import annotations

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import collections.abc

import _zivid


class CameraState:
    """Information about camera connection state, temperatures, etc."""

    class Network:
        """Current network state"""

        class LocalInterface:
            """Current state of the computer's local network interface."""

            class Ethernet:
                """Current Ethernet state"""

                class LinkSpeed:
                    """The link speed that the computer's Ethernet interface is operating at."""

                    link100Mbps = "link100Mbps"
                    link10Gbps = "link10Gbps"
                    link10Mbps = "link10Mbps"
                    link1Gbps = "link1Gbps"
                    link2_5Gbps = "link2_5Gbps"
                    link5Gbps = "link5Gbps"
                    unknown = "unknown"

                    _valid_values = {
                        "link100Mbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link100Mbps,
                        "link10Gbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link10Gbps,
                        "link10Mbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link10Mbps,
                        "link1Gbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link1Gbps,
                        "link2_5Gbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link2_5Gbps,
                        "link5Gbps": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.link5Gbps,
                        "unknown": _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.unknown,
                    }

                    @classmethod
                    def valid_values(cls):
                        return list(cls._valid_values.keys())

                def __init__(
                    self,
                    link_speed: str = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed().value,
                ):

                    if isinstance(link_speed, _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.enum):
                        self._link_speed = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed(link_speed)
                    elif isinstance(link_speed, str):
                        self._link_speed = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed(
                            self.LinkSpeed._valid_values[link_speed]
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str, got {value_type}".format(value_type=type(link_speed))
                        )

                @property
                def link_speed(self):
                    """The link speed that the computer's Ethernet interface is operating at."""
                    if self._link_speed.value is None:
                        return None
                    for key, internal_value in self.LinkSpeed._valid_values.items():
                        if internal_value == self._link_speed.value:
                            return key
                    raise ValueError("Unsupported value {value}".format(value=self._link_speed))

                @link_speed.setter
                def link_speed(self, value):
                    if isinstance(value, str):
                        self._link_speed = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed(
                            self.LinkSpeed._valid_values[value]
                        )
                    elif isinstance(value, _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed.enum):
                        self._link_speed = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed(value)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str, got {value_type}".format(value_type=type(value))
                        )

                def __eq__(self, other):
                    if self._link_speed == other._link_speed:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_camera_state_network_local_interface_ethernet(self))

            class IPV4:
                """Current IPv4 protocol state of the computer's local network interface."""

                class Subnet:
                    """An IPv4 subnet that the local network interface is connected to."""

                    def __init__(
                        self,
                        address: str = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Address().value,
                        mask: str = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Mask().value,
                    ):

                        if isinstance(address, (str,)):
                            self._address = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Address(address)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(address))
                            )

                        if isinstance(mask, (str,)):
                            self._mask = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Mask(mask)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(mask))
                            )

                    @property
                    def address(self):
                        """IP address of the computer's local network interface."""
                        return self._address.value

                    @property
                    def mask(self):
                        """Subnet mask of the computer's local network interface."""
                        return self._mask.value

                    @address.setter
                    def address(self, value):
                        if isinstance(value, (str,)):
                            self._address = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Address(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str, got {value_type}".format(value_type=type(value))
                            )

                    @mask.setter
                    def mask(self, value):
                        if isinstance(value, (str,)):
                            self._mask = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Mask(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str, got {value_type}".format(value_type=type(value))
                            )

                    def __eq__(self, other):
                        if self._address == other._address and self._mask == other._mask:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_camera_state_network_local_interface_ipv4_subnet(self))

                def __init__(
                    self,
                    subnets=None,
                ):

                    if subnets is None:
                        self._subnets = []
                    elif isinstance(subnets, (collections.abc.Iterable,)):
                        self._subnets = []
                        for item in subnets:
                            if isinstance(item, self.Subnet):
                                self._subnets.append(item)
                            else:
                                raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (collections.abc.Iterable,) or None, got {value_type}".format(
                                value_type=type(subnets)
                            )
                        )

                @property
                def subnets(self):
                    """List of IPv4 addresses and subnet masks that the interface is configured with. This list can contain multiple entries if the local network interface is configured with multiple IPv4 addresses."""
                    return self._subnets

                @subnets.setter
                def subnets(self, value):
                    if not isinstance(value, (collections.abc.Iterable,)):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._subnets = []
                    for item in value:
                        if isinstance(item, self.Subnet):
                            self._subnets.append(item)
                        else:
                            raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))

                def __eq__(self, other):
                    if self._subnets == other._subnets:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_camera_state_network_local_interface_ipv4(self))

            def __init__(
                self,
                interface_name: str = _zivid.CameraState.Network.LocalInterface.InterfaceName().value,
                ethernet: CameraState.Network.LocalInterface.Ethernet | None = None,
                ipv4: CameraState.Network.LocalInterface.IPV4 | None = None,
            ):

                if isinstance(interface_name, (str,)):
                    self._interface_name = _zivid.CameraState.Network.LocalInterface.InterfaceName(interface_name)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(interface_name))
                    )

                if ethernet is None:
                    ethernet = self.Ethernet()
                if not isinstance(ethernet, self.Ethernet):
                    raise TypeError("Unsupported type: {value}".format(value=type(ethernet)))
                self._ethernet = ethernet

                if ipv4 is None:
                    ipv4 = self.IPV4()
                if not isinstance(ipv4, self.IPV4):
                    raise TypeError("Unsupported type: {value}".format(value=type(ipv4)))
                self._ipv4 = ipv4

            @property
            def interface_name(self):
                """Name of the computer's local network interface."""
                return self._interface_name.value

            @property
            def ethernet(self):
                """Current Ethernet state"""
                return self._ethernet

            @property
            def ipv4(self):
                """Current IPv4 protocol state of the computer's local network interface."""
                return self._ipv4

            @interface_name.setter
            def interface_name(self, value):
                if isinstance(value, (str,)):
                    self._interface_name = _zivid.CameraState.Network.LocalInterface.InterfaceName(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            @ethernet.setter
            def ethernet(self, value):
                if not isinstance(value, self.Ethernet):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._ethernet = value

            @ipv4.setter
            def ipv4(self, value):
                if not isinstance(value, self.IPV4):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._ipv4 = value

            def __eq__(self, other):
                if (
                    self._interface_name == other._interface_name
                    and self._ethernet == other._ethernet
                    and self._ipv4 == other._ipv4
                ):
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_state_network_local_interface(self))

        class Ethernet:
            """Current Ethernet state"""

            class LinkSpeed:
                """The link speed that the camera's Ethernet interface is operating at."""

                link100Mbps = "link100Mbps"
                link10Gbps = "link10Gbps"
                link10Mbps = "link10Mbps"
                link1Gbps = "link1Gbps"
                link2_5Gbps = "link2_5Gbps"
                link5Gbps = "link5Gbps"
                unknown = "unknown"

                _valid_values = {
                    "link100Mbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link100Mbps,
                    "link10Gbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link10Gbps,
                    "link10Mbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link10Mbps,
                    "link1Gbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link1Gbps,
                    "link2_5Gbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link2_5Gbps,
                    "link5Gbps": _zivid.CameraState.Network.Ethernet.LinkSpeed.link5Gbps,
                    "unknown": _zivid.CameraState.Network.Ethernet.LinkSpeed.unknown,
                }

                @classmethod
                def valid_values(cls):
                    return list(cls._valid_values.keys())

            def __init__(
                self,
                link_speed: str = _zivid.CameraState.Network.Ethernet.LinkSpeed().value,
            ):

                if isinstance(link_speed, _zivid.CameraState.Network.Ethernet.LinkSpeed.enum):
                    self._link_speed = _zivid.CameraState.Network.Ethernet.LinkSpeed(link_speed)
                elif isinstance(link_speed, str):
                    self._link_speed = _zivid.CameraState.Network.Ethernet.LinkSpeed(
                        self.LinkSpeed._valid_values[link_speed]
                    )
                else:
                    raise TypeError(
                        "Unsupported type, expected: str, got {value_type}".format(value_type=type(link_speed))
                    )

            @property
            def link_speed(self):
                """The link speed that the camera's Ethernet interface is operating at."""
                if self._link_speed.value is None:
                    return None
                for key, internal_value in self.LinkSpeed._valid_values.items():
                    if internal_value == self._link_speed.value:
                        return key
                raise ValueError("Unsupported value {value}".format(value=self._link_speed))

            @link_speed.setter
            def link_speed(self, value):
                if isinstance(value, str):
                    self._link_speed = _zivid.CameraState.Network.Ethernet.LinkSpeed(
                        self.LinkSpeed._valid_values[value]
                    )
                elif isinstance(value, _zivid.CameraState.Network.Ethernet.LinkSpeed.enum):
                    self._link_speed = _zivid.CameraState.Network.Ethernet.LinkSpeed(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            def __eq__(self, other):
                if self._link_speed == other._link_speed:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_state_network_ethernet(self))

        class IPV4:
            """Current IPv4 protocol state"""

            def __init__(
                self,
                address: str = _zivid.CameraState.Network.IPV4.Address().value,
            ):

                if isinstance(address, (str,)):
                    self._address = _zivid.CameraState.Network.IPV4.Address(address)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (str,), got {value_type}".format(value_type=type(address))
                    )

            @property
            def address(self):
                """Current IPv4 address"""
                return self._address.value

            @address.setter
            def address(self, value):
                if isinstance(value, (str,)):
                    self._address = _zivid.CameraState.Network.IPV4.Address(value)
                else:
                    raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

            def __eq__(self, other):
                if self._address == other._address:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_camera_state_network_ipv4(self))

        def __init__(
            self,
            local_interfaces=None,
            ethernet: CameraState.Network.Ethernet | None = None,
            ipv4: CameraState.Network.IPV4 | None = None,
        ):

            if local_interfaces is None:
                self._local_interfaces = []
            elif isinstance(local_interfaces, (collections.abc.Iterable,)):
                self._local_interfaces = []
                for item in local_interfaces:
                    if isinstance(item, self.LocalInterface):
                        self._local_interfaces.append(item)
                    else:
                        raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))
            else:
                raise TypeError(
                    "Unsupported type, expected: (collections.abc.Iterable,) or None, got {value_type}".format(
                        value_type=type(local_interfaces)
                    )
                )

            if ethernet is None:
                ethernet = self.Ethernet()
            if not isinstance(ethernet, self.Ethernet):
                raise TypeError("Unsupported type: {value}".format(value=type(ethernet)))
            self._ethernet = ethernet

            if ipv4 is None:
                ipv4 = self.IPV4()
            if not isinstance(ipv4, self.IPV4):
                raise TypeError("Unsupported type: {value}".format(value=type(ipv4)))
            self._ipv4 = ipv4

        @property
        def local_interfaces(self):
            """List of the computer's local network interfaces that discovered the camera. In the most common scenario with the camera connected to the computer with an Ethernet cable, this list will contain only the network interface for that Ethernet port. In more complex network scenarios it can be the case that the camera is discovered by multiple interfaces, and in that case this list will contain multiple network interfaces. However, when `CameraState::Status` is `connected`, only the one network interface that has the active TCP/IP connection to the camera will be listed."""
            return self._local_interfaces

        @property
        def ethernet(self):
            """Current Ethernet state"""
            return self._ethernet

        @property
        def ipv4(self):
            """Current IPv4 protocol state"""
            return self._ipv4

        @local_interfaces.setter
        def local_interfaces(self, value):
            if not isinstance(value, (collections.abc.Iterable,)):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._local_interfaces = []
            for item in value:
                if isinstance(item, self.LocalInterface):
                    self._local_interfaces.append(item)
                else:
                    raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))

        @ethernet.setter
        def ethernet(self, value):
            if not isinstance(value, self.Ethernet):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._ethernet = value

        @ipv4.setter
        def ipv4(self, value):
            if not isinstance(value, self.IPV4):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._ipv4 = value

        def __eq__(self, other):
            if (
                self._local_interfaces == other._local_interfaces
                and self._ethernet == other._ethernet
                and self._ipv4 == other._ipv4
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_camera_state_network(self))

    class Temperature:
        """Current temperature(s)"""

        def __init__(
            self,
            dmd: float | int = _zivid.CameraState.Temperature.DMD().value,
            general: float | int = _zivid.CameraState.Temperature.General().value,
            led: float | int = _zivid.CameraState.Temperature.LED().value,
            lens: float | int = _zivid.CameraState.Temperature.Lens().value,
            pcb: float | int = _zivid.CameraState.Temperature.PCB().value,
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
                    "Unsupported type, expected: (float, int,), got {value_type}".format(value_type=type(dmd))
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
                    "Unsupported type, expected: (float, int,), got {value_type}".format(value_type=type(general))
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
                    "Unsupported type, expected: (float, int,), got {value_type}".format(value_type=type(led))
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
                    "Unsupported type, expected: (float, int,), got {value_type}".format(value_type=type(lens))
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
                    "Unsupported type, expected: (float, int,), got {value_type}".format(value_type=type(pcb))
                )

        @property
        def dmd(self):
            """DMD temperature"""
            return self._dmd.value

        @property
        def general(self):
            """General temperature"""
            return self._general.value

        @property
        def led(self):
            """LED temperature"""
            return self._led.value

        @property
        def lens(self):
            """Lens temperature"""
            return self._lens.value

        @property
        def pcb(self):
            """PCB temperature"""
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
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int, got {value_type}".format(value_type=type(value))
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
        """If the camera status is `inaccessible`, then this enum value will give you the reason."""

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
        """This enum describes the current status of this camera. The enum can have the following values: * `inaccessible`: The camera was discovered, but the SDK is not able to connect to this camera. This can be because the IP settings of the camera and the PC are not compatible, or because there are several cameras with the same IP connected to your PC. The `InaccessibleReason` enum will give you more details about why this camera is not accessible. The network configuration of the camera can be changed using the ZividNetworkCameraConfigurator CLI tool. See the knowledge base for more information. * `busy`: The camera is currently in use by another process. This can be a different process on the same PC, or on a different PC if the camera is shared on a network. * `applyingNetworkConfiguration`: The camera network configuration is being changed by the current process. * `firmwareUpdateRequired`: The camera is accessible, but requires a firmware update before you can connect to it. * `updatingFirmware`: The camera firmware is currently being updated in the current process. * `available`: The camera is available for use by the current process. This means that you can invoke camera.connect() on this camera. * `connecting`: The camera is currently connecting in the current process. * `connected`: The camera is connected in the current process. This means camera.connect() has successfully completed and you can capture using this camera. * `disconnecting`: The camera is currently disconnecting in the current process. When disconnection has completed, the camera will normally go back to the `available` state. * `disappeared`: The camera was found earlier, but it can no longer be found. The connection between the PC and the camera may be disrupted, or the camera may have lost power. When in `disappeared` state, the camera will not be returned from `Application::cameras()`. The camera will go back to one of the other states if the camera is later found again."""

        applyingNetworkConfiguration = "applyingNetworkConfiguration"
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
            "applyingNetworkConfiguration": _zivid.CameraState.Status.applyingNetworkConfiguration,
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
        available: bool = _zivid.CameraState.Available().value,
        connected: bool = _zivid.CameraState.Connected().value,
        inaccessible_reason: str | None = _zivid.CameraState.InaccessibleReason().value,
        status: str = _zivid.CameraState.Status().value,
        network: CameraState.Network | None = None,
        temperature: CameraState.Temperature | None = None,
    ):

        if isinstance(available, (bool,)):
            self._available = _zivid.CameraState.Available(available)
        else:
            raise TypeError("Unsupported type, expected: (bool,), got {value_type}".format(value_type=type(available)))

        if isinstance(connected, (bool,)):
            self._connected = _zivid.CameraState.Connected(connected)
        else:
            raise TypeError("Unsupported type, expected: (bool,), got {value_type}".format(value_type=type(connected)))

        if isinstance(inaccessible_reason, _zivid.CameraState.InaccessibleReason.enum) or inaccessible_reason is None:
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(inaccessible_reason)
        elif isinstance(inaccessible_reason, str):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(
                self.InaccessibleReason._valid_values[inaccessible_reason]
            )
        else:
            raise TypeError(
                "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(inaccessible_reason))
            )

        if isinstance(status, _zivid.CameraState.Status.enum):
            self._status = _zivid.CameraState.Status(status)
        elif isinstance(status, str):
            self._status = _zivid.CameraState.Status(self.Status._valid_values[status])
        else:
            raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(status)))

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
        """Flag if camera is physically connected to the computer and is available for use, but not connected in software. This corresponds to the Status enums `available` or `firmwareUpdateRequired`. Zivid recommends to use the Status enum instead of this bool."""
        return self._available.value

    @property
    def connected(self):
        """Flag if camera is connected in software. This bool is true when the Status value is `connected`. Zivid recommends to use the Status enum instead of this bool."""
        return self._connected.value

    @property
    def inaccessible_reason(self):
        """If the camera status is `inaccessible`, then this enum value will give you the reason."""
        if self._inaccessible_reason.value is None:
            return None
        for key, internal_value in self.InaccessibleReason._valid_values.items():
            if internal_value == self._inaccessible_reason.value:
                return key
        raise ValueError("Unsupported value {value}".format(value=self._inaccessible_reason))

    @property
    def status(self):
        """This enum describes the current status of this camera. The enum can have the following values: * `inaccessible`: The camera was discovered, but the SDK is not able to connect to this camera. This can be because the IP settings of the camera and the PC are not compatible, or because there are several cameras with the same IP connected to your PC. The `InaccessibleReason` enum will give you more details about why this camera is not accessible. The network configuration of the camera can be changed using the ZividNetworkCameraConfigurator CLI tool. See the knowledge base for more information. * `busy`: The camera is currently in use by another process. This can be a different process on the same PC, or on a different PC if the camera is shared on a network. * `applyingNetworkConfiguration`: The camera network configuration is being changed by the current process. * `firmwareUpdateRequired`: The camera is accessible, but requires a firmware update before you can connect to it. * `updatingFirmware`: The camera firmware is currently being updated in the current process. * `available`: The camera is available for use by the current process. This means that you can invoke camera.connect() on this camera. * `connecting`: The camera is currently connecting in the current process. * `connected`: The camera is connected in the current process. This means camera.connect() has successfully completed and you can capture using this camera. * `disconnecting`: The camera is currently disconnecting in the current process. When disconnection has completed, the camera will normally go back to the `available` state. * `disappeared`: The camera was found earlier, but it can no longer be found. The connection between the PC and the camera may be disrupted, or the camera may have lost power. When in `disappeared` state, the camera will not be returned from `Application::cameras()`. The camera will go back to one of the other states if the camera is later found again."""
        if self._status.value is None:
            return None
        for key, internal_value in self.Status._valid_values.items():
            if internal_value == self._status.value:
                return key
        raise ValueError("Unsupported value {value}".format(value=self._status))

    @property
    def network(self):
        """Current network state"""
        return self._network

    @property
    def temperature(self):
        """Current temperature(s)"""
        return self._temperature

    @available.setter
    def available(self, value):
        if isinstance(value, (bool,)):
            self._available = _zivid.CameraState.Available(value)
        else:
            raise TypeError("Unsupported type, expected: bool, got {value_type}".format(value_type=type(value)))

    @connected.setter
    def connected(self, value):
        if isinstance(value, (bool,)):
            self._connected = _zivid.CameraState.Connected(value)
        else:
            raise TypeError("Unsupported type, expected: bool, got {value_type}".format(value_type=type(value)))

    @inaccessible_reason.setter
    def inaccessible_reason(self, value):
        if isinstance(value, str):
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(
                self.InaccessibleReason._valid_values[value]
            )
        elif isinstance(value, _zivid.CameraState.InaccessibleReason.enum) or value is None:
            self._inaccessible_reason = _zivid.CameraState.InaccessibleReason(value)
        else:
            raise TypeError("Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value)))

    @status.setter
    def status(self, value):
        if isinstance(value, str):
            self._status = _zivid.CameraState.Status(self.Status._valid_values[value])
        elif isinstance(value, _zivid.CameraState.Status.enum):
            self._status = _zivid.CameraState.Status(value)
        else:
            raise TypeError("Unsupported type, expected: str, got {value_type}".format(value_type=type(value)))

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

    @classmethod
    def from_serialized(cls, value):
        return _to_camera_state(_zivid.CameraState.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_camera_state(self).serialize()

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

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_camera_state(_to_internal_camera_state(self))


def _to_camera_state_network_local_interface_ethernet(internal_ethernet):
    return CameraState.Network.LocalInterface.Ethernet(
        link_speed=internal_ethernet.link_speed.value,
    )


def _to_camera_state_network_local_interface_ipv4_subnet(internal_subnet):
    return CameraState.Network.LocalInterface.IPV4.Subnet(
        address=internal_subnet.address.value,
        mask=internal_subnet.mask.value,
    )


def _to_camera_state_network_local_interface_ipv4(internal_ipv4):
    return CameraState.Network.LocalInterface.IPV4(
        subnets=[_to_camera_state_network_local_interface_ipv4_subnet(value) for value in internal_ipv4.subnets.value],
    )


def _to_camera_state_network_local_interface(internal_local_interface):
    return CameraState.Network.LocalInterface(
        ethernet=_to_camera_state_network_local_interface_ethernet(internal_local_interface.ethernet),
        ipv4=_to_camera_state_network_local_interface_ipv4(internal_local_interface.ipv4),
        interface_name=internal_local_interface.interface_name.value,
    )


def _to_camera_state_network_ethernet(internal_ethernet):
    return CameraState.Network.Ethernet(
        link_speed=internal_ethernet.link_speed.value,
    )


def _to_camera_state_network_ipv4(internal_ipv4):
    return CameraState.Network.IPV4(
        address=internal_ipv4.address.value,
    )


def _to_camera_state_network(internal_network):
    return CameraState.Network(
        local_interfaces=[
            _to_camera_state_network_local_interface(value) for value in internal_network.local_interfaces.value
        ],
        ethernet=_to_camera_state_network_ethernet(internal_network.ethernet),
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


def _to_internal_camera_state_network_local_interface_ethernet(ethernet):
    internal_ethernet = _zivid.CameraState.Network.LocalInterface.Ethernet()

    internal_ethernet.link_speed = _zivid.CameraState.Network.LocalInterface.Ethernet.LinkSpeed(
        ethernet._link_speed.value
    )

    return internal_ethernet


def _to_internal_camera_state_network_local_interface_ipv4_subnet(subnet):
    internal_subnet = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet()

    internal_subnet.address = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Address(subnet.address)
    internal_subnet.mask = _zivid.CameraState.Network.LocalInterface.IPV4.Subnet.Mask(subnet.mask)

    return internal_subnet


def _to_internal_camera_state_network_local_interface_ipv4(ipv4):
    internal_ipv4 = _zivid.CameraState.Network.LocalInterface.IPV4()

    temp_subnets = _zivid.CameraState.Network.LocalInterface.IPV4.Subnets()
    for value in ipv4.subnets:
        temp_subnets.append(_to_internal_camera_state_network_local_interface_ipv4_subnet(value))
    internal_ipv4.subnets = temp_subnets

    return internal_ipv4


def _to_internal_camera_state_network_local_interface(local_interface):
    internal_local_interface = _zivid.CameraState.Network.LocalInterface()

    internal_local_interface.interface_name = _zivid.CameraState.Network.LocalInterface.InterfaceName(
        local_interface.interface_name
    )

    internal_local_interface.ethernet = _to_internal_camera_state_network_local_interface_ethernet(
        local_interface.ethernet
    )
    internal_local_interface.ipv4 = _to_internal_camera_state_network_local_interface_ipv4(local_interface.ipv4)
    return internal_local_interface


def _to_internal_camera_state_network_ethernet(ethernet):
    internal_ethernet = _zivid.CameraState.Network.Ethernet()

    internal_ethernet.link_speed = _zivid.CameraState.Network.Ethernet.LinkSpeed(ethernet._link_speed.value)

    return internal_ethernet


def _to_internal_camera_state_network_ipv4(ipv4):
    internal_ipv4 = _zivid.CameraState.Network.IPV4()

    internal_ipv4.address = _zivid.CameraState.Network.IPV4.Address(ipv4.address)

    return internal_ipv4


def _to_internal_camera_state_network(network):
    internal_network = _zivid.CameraState.Network()

    temp_local_interfaces = _zivid.CameraState.Network.LocalInterfaces()
    for value in network.local_interfaces:
        temp_local_interfaces.append(_to_internal_camera_state_network_local_interface(value))
    internal_network.local_interfaces = temp_local_interfaces

    internal_network.ethernet = _to_internal_camera_state_network_ethernet(network.ethernet)
    internal_network.ipv4 = _to_internal_camera_state_network_ipv4(network.ipv4)
    return internal_network


def _to_internal_camera_state_temperature(temperature):
    internal_temperature = _zivid.CameraState.Temperature()

    internal_temperature.dmd = _zivid.CameraState.Temperature.DMD(temperature.dmd)
    internal_temperature.general = _zivid.CameraState.Temperature.General(temperature.general)
    internal_temperature.led = _zivid.CameraState.Temperature.LED(temperature.led)
    internal_temperature.lens = _zivid.CameraState.Temperature.Lens(temperature.lens)
    internal_temperature.pcb = _zivid.CameraState.Temperature.PCB(temperature.pcb)

    return internal_temperature


def _to_internal_camera_state(camera_state):
    internal_camera_state = _zivid.CameraState()

    internal_camera_state.available = _zivid.CameraState.Available(camera_state.available)
    internal_camera_state.connected = _zivid.CameraState.Connected(camera_state.connected)
    internal_camera_state.inaccessible_reason = _zivid.CameraState.InaccessibleReason(
        camera_state._inaccessible_reason.value
    )
    internal_camera_state.status = _zivid.CameraState.Status(camera_state._status.value)

    internal_camera_state.network = _to_internal_camera_state_network(camera_state.network)
    internal_camera_state.temperature = _to_internal_camera_state_temperature(camera_state.temperature)
    return internal_camera_state
