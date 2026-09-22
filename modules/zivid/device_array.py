"""Contains the DeviceArray class."""

from __future__ import annotations

import _zivid
import numpy


def _require_stream_or_queue(stream_or_queue):
    """Validate that the argument is a CUDAStreamPtr, OpenCLCommandQueuePtr, or StreamOrQueue."""
    if not isinstance(stream_or_queue, (_zivid.CUDAStreamPtr, _zivid.OpenCLCommandQueuePtr, _zivid.StreamOrQueue)):
        raise TypeError(
            "Expected an instance of CUDAStreamPtr, OpenCLCommandQueuePtr, or StreamOrQueue, got {}".format(
                type(stream_or_queue)
            )
        )


_CUDA_ARRAY_INTERFACE_TYPESTR = {
    _zivid.ImageDeviceArrayRGBA: "|u1",
    _zivid.ImageDeviceArrayBGRA: "|u1",
    _zivid.ImageDeviceArrayRGBA_SRGB: "|u1",
    _zivid.ImageDeviceArrayBGRA_SRGB: "|u1",
    _zivid.ImageDeviceArrayRGBAf: "<f4",
    _zivid.ImageDeviceArrayRGB: "|u1",
    _zivid.ImageDeviceArrayRGB_SRGB: "|u1",
    _zivid.ImageDeviceArrayBGR: "|u1",
    _zivid.ImageDeviceArrayBGR_SRGB: "|u1",
    _zivid.DeviceArrayPointXYZ: "<f4",
    _zivid.DeviceArrayPointXYZW: "<f4",
    _zivid.DeviceArrayPointZ: "<f4",
    _zivid.DeviceArraySNR: "<f4",
    _zivid.DeviceArrayNormalXYZ: "<f4",
}


class DeviceArray:
    """A reference-counted handle to data on a GPU device.

    The underlying device buffer is reference counted and will remain valid as long as any
    DeviceArray instance references it. SDK-created DeviceArrays are bound to the lifetime of
    the Zivid Application that produced them — they must not outlive it. For device memory
    that needs to outlive the Application, allocate it yourself and pass it into the fill
    variant of the acquisition method (not yet exposed at the Python layer).

    A DeviceArray is already synchronized against the stream/queue that was passed to the
    acquisition method (for example ``Frame2D.image_device_array(stream_or_queue, color_format)``),
    so ``device_pointer()`` is a plain accessor that does no additional sync.
    """

    def __init__(self, impl):
        """Initialize DeviceArray wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        allowed_types = (
            _zivid.ImageDeviceArrayRGBA,
            _zivid.ImageDeviceArrayBGRA,
            _zivid.ImageDeviceArrayRGBA_SRGB,
            _zivid.ImageDeviceArrayBGRA_SRGB,
            _zivid.ImageDeviceArrayRGBAf,
            _zivid.ImageDeviceArrayRGB,
            _zivid.ImageDeviceArrayRGB_SRGB,
            _zivid.ImageDeviceArrayBGR,
            _zivid.ImageDeviceArrayBGR_SRGB,
            _zivid.DeviceArrayPointXYZ,
            _zivid.DeviceArrayPointXYZW,
            _zivid.DeviceArrayPointZ,
            _zivid.DeviceArraySNR,
            _zivid.DeviceArrayNormalXYZ,
        )
        if not isinstance(impl, allowed_types):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected one of {}".format(
                    type(impl), ", ".join(allowed_types)
                ),
            )
        self.__impl = impl

    @property
    def _impl(self):
        """Internal access to the binding-level handle. For zivid module use only."""
        return self.__impl

    @property
    def shape(self) -> tuple:
        """Get the shape of the device array.

        Returns:
            A tuple representing the shape of the array.
        """
        return self.__impl.shape

    @property
    def strides(self) -> tuple:
        """Get the strides of the device array in element count.

        Returns:
            A tuple representing the strides of the array in element count.
        """
        return self.__impl.strides

    @property
    def strides_in_bytes(self) -> tuple:
        """Get the strides of the device array in bytes.

        Returns:
            A tuple representing the strides of the array in bytes.
        """
        return self.__impl.strides_in_bytes

    @property
    def size_bytes(self) -> int:
        """Get the size of the device array in bytes.

        Returns:
            The size of the array in bytes.
        """
        return self.__impl.size_bytes

    @property
    def backend(self):
        """Get the backend of the device array.

        Returns:
            A string representing the backend of the array.
        """
        return self.__impl.backend

    @property
    def is_valid(self) -> bool:
        """Check if the device array is valid.

        Returns:
            True if the array is valid, False otherwise.
        """
        return self.__impl.is_valid

    @property
    def is_empty(self) -> bool:
        """Check if the device array is empty.

        Returns:
            True if the array is empty, False otherwise.
        """
        return self.__impl.is_empty

    def device_pointer(self) -> int:
        """Get the raw device pointer for the array data.

        No synchronization is performed. The DeviceArray was already synchronized against the
        caller's stream or queue at acquisition time (by the method that produced it, e.g.
        ``Frame2D.image_device_array(stream_or_queue, color_format)``). Consumer work can be enqueued
        on that same stream or queue immediately.

        Returns:
            An integer representing the device pointer to the array data (``cudaPtr`` for
            CUDA builds, ``cl_mem`` cast to an integer for OpenCL builds).
        """
        return self.__impl.device_pointer()

    def __require_cuda_backend(self, protocol_member: str) -> None:
        if self.__impl.backend != _zivid.ComputeBackend.cuda:
            raise AttributeError(
                "{} is only available on the CUDA backend (this DeviceArray uses the {} backend); use "
                "device_pointer(), which is a cl_mem handle on OpenCL.".format(protocol_member, self.__impl.backend)
            )

    @property
    def __cuda_array_interface__(self):
        """Expose the GPU buffer to CUDA array libraries (CuPy, PyTorch, Numba) zero-copy.

        Returns the CUDA Array Interface (version 3) describing this array's device memory, so that
        ``cupy.asarray(device_array)`` and ``torch.as_tensor(device_array, device="cuda")`` work
        directly without manually threading the device pointer, shape and strides. As with the other
        accessors, the buffer is only valid once the stream or queue it was acquired with has been
        synchronized (e.g. via ``zivid.synchronize_stream``); the caller must do so before the
        consumer reads it.

        Only available on the CUDA backend. On the OpenCL backend ``device_pointer()`` returns a
        ``cl_mem`` handle rather than a CUDA device address, which the CUDA Array Interface cannot
        represent, so this attribute is absent there.

        Raises:
            AttributeError: If the backend is not CUDA, or the format does not expose a typed GPU buffer.
        """
        self.__require_cuda_backend("__cuda_array_interface__")
        try:
            typestr = _CUDA_ARRAY_INTERFACE_TYPESTR[type(self.__impl)]
        except KeyError:
            raise AttributeError("__cuda_array_interface__ is not available for {}".format(type(self.__impl))) from None
        return {
            "shape": tuple(self.__impl.shape),
            "typestr": typestr,
            "data": (self.__impl.device_pointer(), False),
            "strides": tuple(self.__impl.strides_in_bytes),
            "version": 3,
        }

    def __dlpack_device__(self) -> tuple:
        """Report which DLPack device this array's buffer lives on.

        Returns:
            A ``(device_type, device_id)`` tuple, where ``device_type`` is kDLCUDA (2) and ``device_id``
            is the CUDA device ordinal the buffer was allocated on.

        Raises:
            AttributeError: If the backend is not CUDA.
        """
        self.__require_cuda_backend("__dlpack_device__")
        return self.__impl.__dlpack_device__()

    def __dlpack__(self, stream=None, max_version=None, dl_device=None, copy=None):  # pylint: disable=unused-argument
        """Expose the GPU buffer to DLPack consumers (PyTorch, CuPy, JAX, cupoch) zero-copy.

        Wraps the device array in a PyCapsule that references its device memory, so that
        ``torch.from_dlpack(device_array)`` and ``cupy.from_dlpack(device_array)`` work directly. The
        DeviceArray is kept alive by the tensor's manager context, so the GPU memory stays valid for
        as long as the consumer holds the imported tensor.

        Args:
            stream: The consumer's CUDA stream, accepted for protocol compatibility and ignored. The
                buffer was already ordered against the stream or queue passed to the acquisition method
                (for example ``Frame2D.image_device_array(stream_or_queue, color_format)``), so no
                further synchronization is performed here.
            max_version: The highest DLPack version the consumer supports, as a ``(major, minor)``
                tuple. Omitting it, or passing ``None``, selects the legacy unversioned exchange.
            dl_device: The ``(device_type, device_id)`` the consumer wants the buffer on. Only the
                CUDA device the buffer already lives on can be produced, so any other device raises
                ``BufferError``.
            copy: Whether the consumer requires its own copy. ``True`` raises ``BufferError``, since
                this hands out a view of memory other Zivid objects still read and the wrapper has no
                GPU copy primitive. ``False`` and ``None`` are honoured, as no copy is ever made.

        Returns:
            A PyCapsule named ``"dltensor_versioned"`` when the consumer accepts DLPack 1.0 or newer,
            otherwise one named ``"dltensor"``.

        Raises:
            AttributeError: If the backend is not CUDA.
            BufferError: If ``dl_device`` names another device, or ``copy`` is True.
        """
        self.__require_cuda_backend("__dlpack__")
        return self.__impl.__dlpack__(max_version=max_version, dl_device=dl_device, copy=copy)

    def copy_to_host_organized_array(self, stream_or_queue) -> numpy.ndarray:
        """Enqueue a device-to-host copy and return a numpy array WITHOUT synchronizing.

        Use this for device arrays obtained from organized point clouds, or from 2D frames.

        Args:
            stream_or_queue: The CUDA stream or OpenCL command queue used when the array was
                acquired. The D2H copy is enqueued on this stream/queue and this function
                returns immediately. The caller is responsible for synchronizing
                ``stream_or_queue`` (e.g. via ``zivid.synchronize_stream``) before reading
                the returned numpy array's data.

        Returns:
            A 3D numpy array whose underlying memory is NOT safe to read until the caller
            synchronizes ``stream_or_queue``.

        Raises:
            TypeError: If the device array type is ImageDeviceArrayRGBAf,
                which does not support copy_to_host_organized_array.

            RuntimeError: If the device array is not from an organized point cloud or 2D frame.

        Notes:
            If you don't want to manage stream synchronization yourself, use the parameterless
            host accessors on Frame2D / PointCloud (e.g. ``frame_2d.image_rgba()``,
            ``point_cloud.copy_data("xyz")``) instead — those return host-ready data using
            the SDK's internal stream.
        """
        if isinstance(self.__impl, _zivid.ImageDeviceArrayRGBAf):
            raise TypeError(
                "copy_to_host_organized_array() is not supported for ImageDeviceArrayRGBAf. "
                "Use device_pointer() for GPU interop instead."
            )

        try:
            return numpy.asarray(self.__impl.to_array_2d(stream_or_queue))
        except RuntimeError as e:
            raise RuntimeError(
                "copy_to_host_organized_array() is only supported for device arrays from organized point clouds"
                " or 2D frames."
            ) from e

    def copy_to_host_unorganized_array(self, stream_or_queue) -> numpy.ndarray:
        """Enqueue a device-to-host copy and return a numpy array WITHOUT synchronizing.

        Use this for device arrays obtained from unorganized point clouds.

        Args:
            stream_or_queue: The CUDA stream or OpenCL command queue used when the array was
                acquired. The D2H copy is enqueued on this stream/queue and this function
                returns immediately. The caller is responsible for synchronizing
                ``stream_or_queue`` (e.g. via ``zivid.synchronize_stream``) before reading
                the returned numpy array's data.

        Raises:
            TypeError: If the device array type is ImageDeviceArrayRGBAf,
                which does not support copy_to_host_unorganized_array.

            RuntimeError: If the device array is not from an unorganized point cloud.

        Returns:
            A numpy array whose underlying memory is NOT safe to read until the caller
            synchronizes ``stream_or_queue``.

        Notes:
            If you don't want to manage stream synchronization yourself, use the parameterless
            host accessors on UnorganizedPointCloud instead.
        """
        if isinstance(self.__impl, _zivid.ImageDeviceArrayRGBAf):
            raise TypeError(
                "copy_to_host_unorganized_array() is not supported for ImageDeviceArrayRGBAf. "
                "Use device_pointer() for GPU interop instead."
            )

        try:
            return numpy.asarray(self.__impl.to_array_1d(stream_or_queue))
        except RuntimeError as e:
            raise RuntimeError(
                "copy_to_host_unorganized_array() is only supported for device arrays from unorganized point clouds."
            ) from e
