"""Contains the CameraAddress class."""

from __future__ import annotations

import _zivid


class CameraAddress:
    """A hostname or IPv4 address identifying a specific Zivid camera for direct connection.

    Use this to connect to a camera at a known network address, bypassing mDNS discovery.
    Accepts either an IPv4 address string (e.g. "172.28.60.5") or a resolvable hostname.
    """

    def __init__(self, value: str):
        """Construct a CameraAddress from a hostname or IPv4 address string.

        Args:
            value: A string containing the hostname or IPv4 address of the camera

        Raises:
            TypeError: If value is not a string
        """
        if not isinstance(value, str):
            raise TypeError("value must be a string")

        self.__impl = _zivid.CameraAddress(value)

    @property
    def value(self) -> str:
        """Get the address value.

        Returns:
            The hostname or IPv4 address string
        """
        return self.__impl.value()

    def __eq__(self, other):
        if not isinstance(other, CameraAddress):
            return False
        return self.__impl == _to_internal_camera_address(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.__impl.to_string()

    def __repr__(self):
        return f"CameraAddress(value={self.value!r})"

    def __hash__(self):
        return hash(self.value)

    def _to_internal(self):
        return self.__impl


def _to_internal_camera_address(address):
    return address._to_internal()  # pylint: disable=protected-access
