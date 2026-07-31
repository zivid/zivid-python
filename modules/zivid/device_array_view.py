"""Contains the DeviceArrayView class and the create_device_array_view factory."""

from __future__ import annotations

import _zivid
import numpy
from zivid.device_array import _require_stream_or_queue
from zivid.pixel_format import PixelFormat, _resolve_color_format


class DeviceArrayView:
    """A non-owning view over caller-owned GPU memory that the Zivid SDK can read or fill.

    Created via ``zivid.create_device_array_view(...)``. The SDK does not take ownership of the underlying buffer; the
    caller must keep the source array alive while the view (and any consumer of it) is in use. Pass to APIs that take a
    DeviceArrayView, such as the ``destination_buffer`` of the ``Frame2D.image_device_array_*`` accessors (a color
    view to fill in place).
    """

    def __init__(self, impl):
        """Initialize DeviceArrayView wrapper.

        This constructor is only used internally; use ``zivid.create_device_array_view(...)`` instead.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        allowed_types = (
            _zivid.DeviceArrayViewRGBA,
            _zivid.DeviceArrayViewBGRA,
            _zivid.DeviceArrayViewRGBA_SRGB,
            _zivid.DeviceArrayViewBGRA_SRGB,
            _zivid.DeviceArrayViewRGBAf,
            _zivid.DeviceArrayViewRGB,
            _zivid.DeviceArrayViewRGB_SRGB,
            _zivid.DeviceArrayViewBGR,
            _zivid.DeviceArrayViewBGR_SRGB,
        )
        if not isinstance(impl, allowed_types):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected one of {}".format(
                    type(impl), ", ".join(str(t) for t in allowed_types)
                ),
            )
        self.__impl = impl

    @property
    def _impl(self):
        """Internal access to the binding-level handle. For zivid module use only."""
        return self.__impl

    @property
    def shape(self) -> tuple:
        """Tuple of view dimensions."""
        return self.__impl.shape

    @property
    def strides(self) -> tuple:
        """Tuple of view strides in element count."""
        return self.__impl.strides

    @property
    def strides_in_bytes(self) -> tuple:
        """Tuple of view strides in bytes."""
        return self.__impl.strides_in_bytes

    @property
    def size_bytes(self) -> int:
        """Size of the view in bytes."""
        return self.__impl.size_bytes

    @property
    def backend(self):
        """Backend of the view (CUDA or OpenCL)."""
        return self.__impl.backend

    @property
    def is_valid(self) -> bool:
        """True if the view is valid (not empty)."""
        return self.__impl.is_valid

    @property
    def is_empty(self) -> bool:
        """True if the view is empty."""
        return self.__impl.is_empty

    def device_pointer(self) -> int:
        """Get the raw device pointer to the viewed memory.

        Returns:
            An integer device pointer (``cudaPtr`` for CUDA builds, ``cl_mem`` cast to an integer
            for OpenCL builds).
        """
        return self.__impl.device_pointer()

    def copy_to_host_organized_array(self, stream_or_queue) -> numpy.ndarray:
        """Enqueue a device-to-host copy and return a numpy array WITHOUT synchronizing.

        Only available for color views. The D2H copy is enqueued on ``stream_or_queue`` and this
        function returns immediately; the caller must synchronize ``stream_or_queue`` (e.g. via
        ``zivid.synchronize_stream``) before reading the returned numpy array.

        Args:
            stream_or_queue: The CUDA stream or OpenCL command queue to enqueue the D2H copy on.

        Returns:
            A 3D numpy array whose memory is NOT safe to read until ``stream_or_queue`` is synchronized.
        """
        return numpy.array(self.__impl.to_array_2d(stream_or_queue))


_PIXEL_FORMAT_FACTORY_METHOD = {
    PixelFormat.RGBA: "create_device_array_view_rgba",
    PixelFormat.BGRA: "create_device_array_view_bgra",
    PixelFormat.RGBA_SRGB: "create_device_array_view_rgba_srgb",
    PixelFormat.BGRA_SRGB: "create_device_array_view_bgra_srgb",
    PixelFormat.RGBAF: "create_device_array_view_rgbaf",
    PixelFormat.RGB: "create_device_array_view_rgb",
    PixelFormat.RGB_SRGB: "create_device_array_view_rgb_srgb",
    PixelFormat.BGR: "create_device_array_view_bgr",
    PixelFormat.BGR_SRGB: "create_device_array_view_bgr_srgb",
}


def _device_array_view_fill_target(destination_buffer, supported):
    """Validate destination_buffer is a fillable DeviceArrayView; return (binding_suffix, impl).

    The color format of a fill is carried by the typed DeviceArrayView, so the binding method is
    selected from the view's type rather than from a PixelFormat argument. ``supported`` is a mapping
    {_zivid.DeviceArrayView* type: binding_suffix} declaring which view types the calling accessor can
    fill. The returned impl is the binding-level handle to pass to the fill method.

    Raises:
        TypeError: If destination_buffer is not a DeviceArrayView that this accessor can fill (e.g. a 3-channel view
            passed to an accessor that only fills 4-channel views).
    """
    impl = destination_buffer._impl  # pylint: disable=protected-access
    suffix = supported.get(type(impl))
    if suffix is None:
        raise TypeError(
            "destination_buffer must be a color DeviceArrayView this accessor can fill, got {}.".format(type(impl))
        )
    return suffix, impl


def create_device_array_view(cuda_array, user_stream, color_format: PixelFormat) -> DeviceArrayView:
    """Wrap a caller-owned CUDA array as a non-owning Zivid DeviceArrayView.

    ``cuda_array`` is any object exposing the CUDA Array Interface (``__cuda_array_interface__``) —
    CuPy ndarrays, PyTorch CUDA tensors, Numba device arrays, etc. The SDK reads from (or fills) the
    underlying buffer; the caller must keep the source array alive until the consumer is done.

    The format is not inferred: pass the ``color_format`` explicitly. A color buffer must be 3D (height, width, 4).

    Args:
        cuda_array:   Source CUDA buffer (any object exposing ``__cuda_array_interface__``).
        user_stream:  A CUDAStreamPtr (or StreamOrQueue) identifying the stream the buffer was
                      produced on. The SDK records an event on it so its reads/writes happen-after
                      the producer's work.
        color_format: A ``zivid.PixelFormat`` selecting the view format.

    Returns:
        A DeviceArrayView.

    Raises:
        TypeError: If ``color_format`` is not a ``zivid.PixelFormat``.
        ValueError: If ``color_format`` is a PixelFormat that create_device_array_view does not
            support (the 4-channel color formats are supported).
    """
    from zivid.application import Application  # pylint: disable=import-outside-toplevel

    factory_method = _resolve_color_format(color_format, _PIXEL_FORMAT_FACTORY_METHOD)
    _require_stream_or_queue(user_stream)
    iface = cuda_array.__cuda_array_interface__
    ptr = int(iface["data"][0])
    shape = tuple(iface["shape"])
    height = int(shape[0])
    width = int(shape[1])
    compute_device = Application().compute_device()
    factory = getattr(compute_device, factory_method)
    return DeviceArrayView(factory(ptr, width, height, user_stream))
