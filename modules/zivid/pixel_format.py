"""Contains the PixelFormat enum used to select a device-array color format."""

from enum import Enum


class PixelFormat(Enum):
    """Color format of a device array or DeviceArrayView.

    The 4-channel formats are 3D (height x width x 4): ``RGBA`` / ``BGRA`` are 8-bit, ``RGBA_SRGB`` / ``BGRA_SRGB``
    are 8-bit sRGB-encoded, and ``RGBAF`` is float32. The 3-channel formats are 3D (height x width x 3, no alpha):
    ``RGB`` / ``BGR`` are 8-bit, ``RGB_SRGB`` / ``BGR_SRGB`` are 8-bit sRGB-encoded.

    Not every accessor accepts every format. ``create_device_array_view`` accepts all color formats; the Frame2D color
    accessors (allocate and fill) cover all color formats, while the point-cloud color accessors cover a subset.
    """

    RGBA = "rgba"
    BGRA = "bgra"
    RGBA_SRGB = "rgba_srgb"
    BGRA_SRGB = "bgra_srgb"
    RGBAF = "rgbaf"
    RGB = "rgb"
    RGB_SRGB = "rgb_srgb"
    BGR = "bgr"
    BGR_SRGB = "bgr_srgb"


def _resolve_color_format(color_format, supported):
    """Validate color_format and map it to an accessor-specific dispatch value.

    Args:
        color_format: The value to validate. Must be a zivid.PixelFormat.
        supported:    An ordered mapping {PixelFormat: dispatch_value} declaring which formats the
                      calling accessor accepts and what to dispatch to for each.

    Returns:
        supported[color_format].

    Raises:
        TypeError: If color_format is not a zivid.PixelFormat.
        ValueError: If color_format is a PixelFormat that this accessor does not support.
    """
    if not isinstance(color_format, PixelFormat):
        raise TypeError("color_format must be a zivid.PixelFormat, got {}".format(type(color_format)))
    if color_format not in supported:
        raise ValueError(
            "{} is not a supported format here. Supported formats: {}.".format(
                color_format.name, ", ".join(fmt.name for fmt in supported)
            )
        )
    return supported[color_format]
