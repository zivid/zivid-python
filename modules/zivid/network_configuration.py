"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import _zivid


class NetworkConfiguration:

    class IPV4:

        class Mode:

            dhcp = "dhcp"
            manual = "manual"

            _valid_values = {
                "dhcp": _zivid.NetworkConfiguration.IPV4.Mode.dhcp,
                "manual": _zivid.NetworkConfiguration.IPV4.Mode.manual,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            address=_zivid.NetworkConfiguration.IPV4.Address().value,
            mode=_zivid.NetworkConfiguration.IPV4.Mode().value,
            subnet_mask=_zivid.NetworkConfiguration.IPV4.SubnetMask().value,
        ):

            if isinstance(address, (str,)):
                self._address = _zivid.NetworkConfiguration.IPV4.Address(address)
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(
                        value_type=type(address)
                    )
                )

            if isinstance(mode, _zivid.NetworkConfiguration.IPV4.Mode.enum):
                self._mode = _zivid.NetworkConfiguration.IPV4.Mode(mode)
            elif isinstance(mode, str):
                self._mode = _zivid.NetworkConfiguration.IPV4.Mode(
                    self.Mode._valid_values[mode]
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(mode)
                    )
                )

            if isinstance(subnet_mask, (str,)):
                self._subnet_mask = _zivid.NetworkConfiguration.IPV4.SubnetMask(
                    subnet_mask
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (str,), got {value_type}".format(
                        value_type=type(subnet_mask)
                    )
                )

        @property
        def address(self):
            return self._address.value

        @property
        def mode(self):
            if self._mode.value is None:
                return None
            for key, internal_value in self.Mode._valid_values.items():
                if internal_value == self._mode.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._mode))

        @property
        def subnet_mask(self):
            return self._subnet_mask.value

        @address.setter
        def address(self, value):
            if isinstance(value, (str,)):
                self._address = _zivid.NetworkConfiguration.IPV4.Address(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @mode.setter
        def mode(self, value):
            if isinstance(value, str):
                self._mode = _zivid.NetworkConfiguration.IPV4.Mode(
                    self.Mode._valid_values[value]
                )
            elif isinstance(value, _zivid.NetworkConfiguration.IPV4.Mode.enum):
                self._mode = _zivid.NetworkConfiguration.IPV4.Mode(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @subnet_mask.setter
        def subnet_mask(self, value):
            if isinstance(value, (str,)):
                self._subnet_mask = _zivid.NetworkConfiguration.IPV4.SubnetMask(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if (
                self._address == other._address
                and self._mode == other._mode
                and self._subnet_mask == other._subnet_mask
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_network_configuration_ipv4(self))

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

    @classmethod
    def load(cls, file_name):
        return _to_network_configuration(_zivid.NetworkConfiguration(str(file_name)))

    def save(self, file_name):
        _to_internal_network_configuration(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_network_configuration(
            _zivid.NetworkConfiguration.from_serialized(str(value))
        )

    def serialize(self):
        return _to_internal_network_configuration(self).serialize()

    def __eq__(self, other):
        if self._ipv4 == other._ipv4:
            return True
        return False

    def __str__(self):
        return str(_to_internal_network_configuration(self))


def _to_network_configuration_ipv4(internal_ipv4):
    return NetworkConfiguration.IPV4(
        address=internal_ipv4.address.value,
        mode=internal_ipv4.mode.value,
        subnet_mask=internal_ipv4.subnet_mask.value,
    )


def _to_network_configuration(internal_network_configuration):
    return NetworkConfiguration(
        ipv4=_to_network_configuration_ipv4(internal_network_configuration.ipv4),
    )


def _to_internal_network_configuration_ipv4(ipv4):
    internal_ipv4 = _zivid.NetworkConfiguration.IPV4()

    internal_ipv4.address = _zivid.NetworkConfiguration.IPV4.Address(ipv4.address)
    internal_ipv4.mode = _zivid.NetworkConfiguration.IPV4.Mode(ipv4._mode.value)
    internal_ipv4.subnet_mask = _zivid.NetworkConfiguration.IPV4.SubnetMask(
        ipv4.subnet_mask
    )

    return internal_ipv4


def _to_internal_network_configuration(network_configuration):
    internal_network_configuration = _zivid.NetworkConfiguration()

    internal_network_configuration.ipv4 = _to_internal_network_configuration_ipv4(
        network_configuration.ipv4
    )
    return internal_network_configuration
