"""Contains the Mask class."""

from __future__ import annotations

import _zivid
import numpy
from zivid.resolution import Resolution


class Mask:
    """Binary mask for filtering point cloud data.

    The Mask class represents a 2D binary mask that can be used to filter point cloud data.
    Non-zero values in the mask indicate that the corresponding point should be masked out
    (set to NaN), while zero values indicate that the point should be preserved.
    """

    def __init__(self, mask_data):
        """Initialize Mask wrapper.

        Args:
            mask_data: Can be one of:
                - A 2D numpy array of booleans or uint8 values
                - A zivid.Resolution object to create a mask that masks out every point
                - An internal _zivid.Mask instance

        Raises:
            TypeError: If argument type is not supported
            ValueError: If numpy array dimensions are incorrect
        """
        if isinstance(mask_data, _zivid.Mask):
            # This constructor is only used internally, and should not be called by the end-user.
            self.__impl = mask_data
        elif hasattr(mask_data, "width") and hasattr(mask_data, "height"):  # Resolution-like object

            if isinstance(mask_data, Resolution):
                self.__impl = _zivid.Mask(mask_data._to_internal())  # pylint: disable=protected-access
            else:
                # Handle other resolution-like objects
                resolution = Resolution(mask_data.width, mask_data.height)
                self.__impl = _zivid.Mask(resolution._to_internal())  # pylint: disable=protected-access
        elif isinstance(mask_data, numpy.ndarray):
            if mask_data.ndim != 2:
                raise ValueError("Mask data must be a 2D array")

            # Convert boolean mask to uint8 (True -> 1, False -> 0)
            if mask_data.dtype == bool:
                mask_array = mask_data.astype(numpy.uint8)
            elif mask_data.dtype == numpy.uint8:
                mask_array = mask_data
            else:
                # Convert other numeric types to uint8, treating non-zero as 1
                mask_array = (mask_data != 0).astype(numpy.uint8)

            height, width = mask_array.shape

            resolution = Resolution(width, height)

            # Create mask from raw data
            mask_bytes = mask_array.tobytes()
            self.__impl = _zivid.Mask(resolution._to_internal(), mask_bytes)  # pylint: disable=protected-access
        else:
            raise TypeError(
                f"Unsupported type for mask_data. Got {type(mask_data)}, expected numpy array or Resolution"
            )

    @property
    def width(self) -> int:
        """Get the width of the mask.

        Returns:
            Width as an integer
        """
        return self.__impl.width()

    @property
    def height(self) -> int:
        """Get the height of the mask.

        Returns:
            Height as an integer
        """
        return self.__impl.height()

    @property
    def resolution(self) -> Resolution:
        """Get the resolution of the mask.

        Returns:
            Resolution object with width and height
        """
        return Resolution(self.__impl.width(), self.__impl.height())

    def to_array(self) -> numpy.ndarray:
        """Convert mask to numpy array.

        Returns:
            2D numpy array of uint8 values
        """
        return numpy.array(self.__impl)

    def __str__(self):
        """Get string representation of the mask."""
        return self.__impl.to_string()

    def __repr__(self):
        """Get string representation of the mask."""
        return self.__impl.to_string()

    def release(self) -> None:
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()

    def __copy__(self):
        return Mask(self.__impl.__copy__())

    def __deepcopy__(self, memodict):
        return Mask(self.__impl.__deepcopy__(memodict))

    def _to_internal(self):
        """Get the internal implementation.

        Returns:
            Internal _zivid.Mask object
        """
        return self.__impl


def _to_internal_mask(mask):
    """Convert Mask to internal implementation.

    Args:
        mask: Mask object to convert

    Returns:
        Internal _zivid.Mask object
    """
    return mask._to_internal()  # pylint: disable=protected-access
