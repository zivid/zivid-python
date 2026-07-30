"""Contains the Resolution class."""

from __future__ import annotations

import _zivid


class Resolution:
    """Class describing a resolution with a width and a height.

    A Resolution represents the dimensions (width x height) of 2D data structures
    such as images, point clouds, or masks.
    """

    def __init__(self, width: int, height: int):
        """Construct a Resolution from width and height.

        Args:
            width: Width as a positive integer
            height: Height as a positive integer

        Raises:
            TypeError: If width or height are not integers
            ValueError: If width or height are not positive
        """
        if not isinstance(width, int) or not isinstance(height, int):
            raise TypeError("Width and height must be integers")

        if width < 0 or height < 0:
            raise ValueError("Width and height must be non-negative")

        self.__impl = _zivid.Resolution(width, height)

    @property
    def width(self) -> int:
        """Get the width value of the resolution.

        Returns:
            Width as an integer
        """
        return self.__impl.width()

    @property
    def height(self) -> int:
        """Get the height value of the resolution.

        Returns:
            Height as an integer
        """
        return self.__impl.height()

    @property
    def size(self) -> int:
        """Get the size (area) that is the product of the width and height.

        Returns:
            Area as an integer (width * height)
        """
        return self.__impl.size()

    def __eq__(self, other):
        """Check if two resolutions are equal.

        Args:
            other: Another Resolution object

        Returns:
            True if width and height are equal, False otherwise
        """
        if not isinstance(other, Resolution):
            return False
        return self.__impl == _to_internal_resolution(other)

    def __ne__(self, other):
        """Check if two resolutions are not equal.

        Args:
            other: Another Resolution object

        Returns:
            True if width or height are different, False otherwise
        """
        return not self.__eq__(other)

    def __str__(self):
        """Get string representation of the resolution.

        Returns:
            String representation like "1920x1080"
        """
        return self.__impl.to_string()

    def __repr__(self):
        """Get string representation of the resolution.

        Returns:
            String representation like "Resolution(width=1920, height=1080)"
        """
        return f"Resolution(width={self.width}, height={self.height})"

    def __hash__(self):
        """Get hash of the resolution for use in sets and dictionaries.

        Returns:
            Hash value based on width and height
        """
        return hash((self.width, self.height))

    def _to_internal(self):
        """Get the internal implementation (for use by other Zivid classes).

        Returns:
            The internal _zivid.Resolution object
        """
        return self.__impl


def _to_internal_resolution(resolution):
    """Convert Resolution to internal implementation.

    Args:
        resolution: Resolution object to convert

    Returns:
        Internal _zivid.Resolution object
    """
    return resolution._to_internal()  # pylint: disable=protected-access
