"""Contains VectorXY class."""

from __future__ import annotations

import _zivid


class VectorXY:
    """Vector with two coordinates as float."""

    def __init__(self, x: float, y: float):
        """Construct a VectorXY object.

        Args:
            x: x coordinate as float
            y: y coordinate as float
        """
        self.__impl = _zivid.VectorXY(x, y)

    @property
    def x(self) -> float:
        """Get the x coordinate.

        Returns:
            The x coordinate as float
        """
        return self.__impl.x

    @x.setter
    def x(self, value: float) -> None:
        """Set the x coordinate.

        Args:
            value: The x coordinate as float
        """
        self.__impl.x = value

    @property
    def y(self) -> float:
        """Get the y coordinate.

        Returns:
            The y coordinate as float
        """
        return self.__impl.y

    @y.setter
    def y(self, value: float) -> None:
        """Set the y coordinate.

        Args:
            value: The y coordinate as float
        """
        self.__impl.y = value

    def angle_in_degrees(self) -> float:
        """Get the direction of this vector as an angle in degrees, in the range [0, 360).

        0 degrees points along the positive x-axis, and the angle increases towards the
        positive y-axis.

        Returns:
            The angle in degrees as float

        Raises:
            RuntimeError: The vector has zero length or is non-finite
        """
        return self.__impl.angle_in_degrees()

    def angle_in_radians(self) -> float:
        """Get the direction of this vector as an angle in radians, in the range [0, 2*pi).

        0 radians points along the positive x-axis, and the angle increases towards the
        positive y-axis.

        Returns:
            The angle in radians as float

        Raises:
            RuntimeError: The vector has zero length or is non-finite
        """
        return self.__impl.angle_in_radians()

    def __eq__(self, other):
        return self.__impl == other._VectorXY__impl

    def __str__(self):
        return str(self.__impl)
