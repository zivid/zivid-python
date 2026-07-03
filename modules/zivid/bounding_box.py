"""Contains BoundingBox class."""

import _zivid


class BoundingBox:
    """Defines a 2D rectangular bounding box in image coordinates."""

    def __init__(self, x, y, width, height):
        """Construct a BoundingBox object.

        Args:
            x: Top-left corner x coordinate as int
            y: Top-left corner y coordinate as int
            width: Width of the bounding box as int
            height: Height of the bounding box as int
        """
        self.__impl = _zivid.BoundingBox(x, y, width, height)

    @property
    def x(self):
        """Get the top-left corner x coordinate.

        Returns:
            The x coordinate as int
        """
        return self.__impl.x

    @x.setter
    def x(self, value):
        """Set the top-left corner x coordinate.

        Args:
            value: The x coordinate as int
        """
        self.__impl.x = value

    @property
    def y(self):
        """Get the top-left corner y coordinate.

        Returns:
            The y coordinate as int
        """
        return self.__impl.y

    @y.setter
    def y(self, value):
        """Set the top-left corner y coordinate.

        Args:
            value: The y coordinate as int
        """
        self.__impl.y = value

    @property
    def width(self):
        """Get the width of the bounding box.

        Returns:
            The width as int
        """
        return self.__impl.width

    @width.setter
    def width(self, value):
        """Set the width of the bounding box.

        Args:
            value: The width as int
        """
        self.__impl.width = value

    @property
    def height(self):
        """Get the height of the bounding box.

        Returns:
            The height as int
        """
        return self.__impl.height

    @height.setter
    def height(self, value):
        """Set the height of the bounding box.

        Args:
            value: The height as int
        """
        self.__impl.height = value

    def __str__(self):
        return str(self.__impl)
