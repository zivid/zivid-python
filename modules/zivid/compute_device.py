"""Contains the GPU compute device, and the handles used to share GPU work with the SDK."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import _zivid

if TYPE_CHECKING:
    import zivid.device_array_view


class ComputeBackend(Enum):
    """GPU compute backend type."""

    cuda = 0
    opencl = 1

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value


def _to_compute_backend(internal_compute_backend):
    return ComputeBackend[internal_compute_backend.name]


class CUDAContextPtr(_zivid.CUDAContextPtr):
    """Wrapper for a user-provided CUDA context (CUcontext).

    Pass it to ``zivid.Application`` to make the SDK use that CUDA context.
    """

    def __init__(self, context: int = 0):  # pylint: disable=useless-parent-delegation
        """Initialize a CUDAContextPtr.

        Args:
            context: The CUcontext handle, as an integer
        """
        super().__init__(context)

    @property
    def context(self) -> int:
        """Get the CUcontext handle.

        Returns:
            The CUcontext handle, as an integer
        """
        return _zivid.CUDAContextPtr.context.__get__(self)

    @context.setter
    def context(self, value: int) -> None:
        _zivid.CUDAContextPtr.context.__set__(self, value)


class CUDAStreamPtr(_zivid.CUDAStreamPtr):
    """Wrapper for a user-provided CUDA stream (CUstream), for synchronization.

    Pass it wherever a function takes a ``stream_or_queue``, to synchronize SDK operations with
    that CUDA stream before device memory is accessed.
    """

    def __init__(self, stream: int = 0):  # pylint: disable=useless-parent-delegation
        """Initialize a CUDAStreamPtr.

        Args:
            stream: The CUstream handle, as an integer. 0 is the CUDA default stream.
        """
        super().__init__(stream)

    @property
    def stream(self) -> int:
        """Get the CUstream handle.

        Returns:
            The CUstream handle, as an integer
        """
        return _zivid.CUDAStreamPtr.stream.__get__(self)

    @stream.setter
    def stream(self, value: int) -> None:
        _zivid.CUDAStreamPtr.stream.__set__(self, value)


class OpenCLCommandQueuePtr(_zivid.OpenCLCommandQueuePtr):
    """Wrapper for a user-provided OpenCL command queue (cl_command_queue), for synchronization.

    Pass it wherever a function takes a ``stream_or_queue``, to synchronize SDK operations with
    that OpenCL command queue before device memory is accessed.
    """

    def __init__(self, command_queue: int = 0):  # pylint: disable=useless-parent-delegation
        """Initialize an OpenCLCommandQueuePtr.

        Args:
            command_queue: The cl_command_queue handle, as an integer
        """
        super().__init__(command_queue)

    @property
    def command_queue(self) -> int:
        """Get the cl_command_queue handle.

        Returns:
            The cl_command_queue handle, as an integer
        """
        return _zivid.OpenCLCommandQueuePtr.command_queue.__get__(self)

    @command_queue.setter
    def command_queue(self, value: int) -> None:
        _zivid.OpenCLCommandQueuePtr.command_queue.__set__(self, value)


class StreamOrQueue(_zivid.StreamOrQueue):
    """A user CUDA stream or OpenCL command queue, selected by which member is populated.

    Functions that take a ``stream_or_queue`` accept a CUDAStreamPtr, an OpenCLCommandQueuePtr or
    a StreamOrQueue. The compute backend of the SDK picks the matching member, and the other must
    be left null, or the call raises:

    * CUDA: ``stream`` is the CUDA stream to synchronize with. A null stream means the CUDA
      default stream. ``command_queue`` must be null.
    * OpenCL: ``command_queue`` is the OpenCL command queue to synchronize with, and must not be
      null. ``stream`` must be null.
    """

    def __init__(
        self,
        stream_or_queue: CUDAStreamPtr | OpenCLCommandQueuePtr | None = None,
        /,
        *,
        stream: CUDAStreamPtr | None = None,
        command_queue: OpenCLCommandQueuePtr | None = None,
    ):
        """Initialize a StreamOrQueue from either a CUDA stream or an OpenCL command queue.

        Pass exactly one of the arguments.

        Args:
            stream_or_queue: A CUDAStreamPtr or an OpenCLCommandQueuePtr
            stream: The CUDA stream, for the CUDA backend
            command_queue: The OpenCL command queue, for the OpenCL backend

        Raises:
            TypeError: If not exactly one argument is given, or if it has the wrong type
        """
        given = [argument for argument in (stream_or_queue, stream, command_queue) if argument is not None]
        if len(given) != 1:
            raise TypeError("Expected exactly one of stream_or_queue, stream and command_queue")
        if stream_or_queue is not None and not isinstance(
            stream_or_queue, (_zivid.CUDAStreamPtr, _zivid.OpenCLCommandQueuePtr)
        ):
            raise TypeError(
                "Unsupported type for stream_or_queue, expected: CUDAStreamPtr or OpenCLCommandQueuePtr, got {}".format(
                    type(stream_or_queue)
                )
            )
        if stream is not None and not isinstance(stream, _zivid.CUDAStreamPtr):
            raise TypeError("Unsupported type for stream, expected: CUDAStreamPtr, got {}".format(type(stream)))
        if command_queue is not None and not isinstance(command_queue, _zivid.OpenCLCommandQueuePtr):
            raise TypeError(
                "Unsupported type for command_queue, expected: OpenCLCommandQueuePtr, got {}".format(
                    type(command_queue)
                )
            )
        super().__init__(given[0])

    @property
    def stream(self) -> CUDAStreamPtr:
        """Get the CUDA stream member.

        Returns:
            The CUDA stream, which is null unless the CUDA backend is selected
        """
        return CUDAStreamPtr(_zivid.StreamOrQueue.stream.__get__(self).stream)

    @stream.setter
    def stream(self, value: CUDAStreamPtr) -> None:
        _zivid.StreamOrQueue.stream.__set__(self, value)

    @property
    def command_queue(self) -> OpenCLCommandQueuePtr:
        """Get the OpenCL command queue member.

        Returns:
            The OpenCL command queue, which is null unless the OpenCL backend is selected
        """
        return OpenCLCommandQueuePtr(_zivid.StreamOrQueue.command_queue.__get__(self).command_queue)

    @command_queue.setter
    def command_queue(self, value: OpenCLCommandQueuePtr) -> None:
        _zivid.StreamOrQueue.command_queue.__set__(self, value)


def _to_stream_or_queue(internal_stream_or_queue):
    command_queue = internal_stream_or_queue.command_queue.command_queue
    if command_queue:
        return StreamOrQueue(OpenCLCommandQueuePtr(command_queue))
    return StreamOrQueue(CUDAStreamPtr(internal_stream_or_queue.stream.stream))


def synchronize_stream(stream_or_queue) -> None:
    """Block until all work previously enqueued on a CUDA stream or OpenCL command queue has completed.

    Equivalent to cudaStreamSynchronize with the CUDA backend and clFinish with the OpenCL backend.

    The host accessors of DeviceArray and DeviceArrayView, such as copy_to_host_organized_array,
    enqueue the device-to-host copy on the given stream or queue and return without waiting for it.
    Call this function on the same stream or queue before reading their results.

    Args:
        stream_or_queue: A CUDAStreamPtr, OpenCLCommandQueuePtr or StreamOrQueue
    """
    _zivid._zividcore.synchronize_stream(stream_or_queue)  # pylint: disable=protected-access


class ComputeDevice:
    """The GPU compute device used by the SDK.

    Get it with ``zivid.Application.compute_device``.
    """

    def __init__(self, impl):
        """Initialize ComputeDevice wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.ComputeDevice):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}.".format(type(impl), _zivid.ComputeDevice)
            )
        self.__impl = impl

    def __str__(self):
        return str(self.__impl)

    @property
    def model(self) -> str:
        """Get the model name of the compute device.

        Returns:
            The model name
        """
        return self.__impl.model

    @property
    def vendor(self) -> str:
        """Get the vendor of the compute device.

        Returns:
            The vendor
        """
        return self.__impl.vendor

    @property
    def backend(self) -> ComputeBackend:
        """Get the compute backend of the compute device.

        Returns:
            The compute backend
        """
        return _to_compute_backend(self.__impl.backend)

    @property
    def cuda_runtime_library_name(self) -> str:
        """Get the shared library name of the CUDA runtime that the SDK was built against.

        This is useful for loading the CUDA runtime with ctypes or similar foreign function
        interfaces for GPU interop, such as CUDA-OpenGL interop. The name is guaranteed to match
        the CUDA version that the SDK was compiled against.

        Returns:
            The platform-specific library name, for example "libcudart.so" on Linux or
            "cudart64_12.dll" on Windows

        Raises:
            RuntimeError: If the compute backend is not CUDA
        """
        return self.__impl.cuda_runtime_library_name

    def native_context(self) -> int:
        """Get the native context handle, for interop with GPU APIs.

        Returns:
            The CUcontext with the CUDA backend, or the cl_context with the OpenCL backend, as an
            integer
        """
        return self.__impl.native_context()

    def native_stream_handle(self) -> int:
        """Get the native stream or queue handle, for advanced GPU interop.

        Use the handle to synchronize with your own GPU operations, to submit custom kernels on the
        same stream, or to check whether the stream has completed. It stays valid for the lifetime
        of the compute device. Do not destroy it.

        With a user-provided context or stream this is the user-provided handle, otherwise the
        handle of the stream or queue the SDK created.

        Returns:
            The CUstream with the CUDA backend, or the cl_command_queue with the OpenCL backend, as
            an integer
        """
        return self.__impl.native_stream_handle()

    def sdk_stream_or_queue(self) -> StreamOrQueue:
        """Get the stream or queue of the SDK, as a backend-agnostic StreamOrQueue.

        A portable alternative to wrapping ``native_stream_handle`` in a CUDAStreamPtr or an
        OpenCLCommandQueuePtr. Pass it to functions that take a ``stream_or_queue`` when you
        do not have a stream or queue of your own. The synchronization is then with the SDK's own
        stream or queue, which costs nothing.

        Returns:
            A StreamOrQueue holding the stream or queue of the SDK
        """
        return _to_stream_or_queue(self.__impl.sdk_stream_or_queue())

    def create_device_array_view_rgba(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the RGBA format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_rgba",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_bgra(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the BGRA format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_bgra",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_rgba_srgb(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the RGBA_SRGB format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_rgba_srgb",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_bgra_srgb(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the BGRA_SRGB format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_bgra_srgb",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_rgbaf(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the RGBAF format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_rgbaf",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_rgb(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the RGB format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_rgb",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_rgb_srgb(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the RGB_SRGB format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_rgb_srgb",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_bgr(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the BGR format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_bgr",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def create_device_array_view_bgr_srgb(
        self, cuda_pointer: int, width: int, height: int, stream_or_queue
    ) -> zivid.device_array_view.DeviceArrayView:
        """Create a DeviceArrayView in the BGR_SRGB format over an externally-owned CUDA device pointer.

        The view does not take ownership of the CUDA memory. The caller must keep the pointer valid for as
        long as the view, and any buffer filled through it, is in use. The device pointer must be created
        using the same CUDA context that the Zivid SDK uses. Only available when the compute backend is CUDA.

        Args:
            cuda_pointer: Non-null CUDA device pointer, as an integer
            width: Width of the image in pixels
            height: Height of the image in pixels
            stream_or_queue: The CUDA stream that the preparation work of the buffer (allocation and any
                data production) was enqueued on

        Returns:
            A DeviceArrayView referencing the provided CUDA memory
        """
        return self.__create_device_array_view(
            "create_device_array_view_bgr_srgb",
            cuda_pointer=cuda_pointer,
            width=width,
            height=height,
            stream_or_queue=stream_or_queue,
        )

    def __create_device_array_view(self, factory, *, cuda_pointer, width, height, stream_or_queue):
        from zivid.device_array import _require_stream_or_queue  # pylint: disable=import-outside-toplevel
        from zivid.device_array_view import DeviceArrayView  # pylint: disable=import-outside-toplevel

        _require_stream_or_queue(stream_or_queue)
        return DeviceArrayView(getattr(self.__impl, factory)(cuda_pointer, width, height, stream_or_queue))
