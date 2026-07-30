"""Contains the Frame2D class."""

from __future__ import annotations

from pathlib import Path

import _zivid
from zivid.camera_info import CameraInfo, _to_camera_info
from zivid.camera_state import CameraState, _to_camera_state
from zivid.device_array import DeviceArray, _require_stream_or_queue
from zivid.device_array_view import DeviceArrayView, _device_array_view_fill_target
from zivid.frame_info import FrameInfo, _to_frame_info
from zivid.image import Image
from zivid.pixel_format import PixelFormat, _resolve_color_format
from zivid.settings2d import Settings2D, _to_settings2d

_COLOR_FORMAT_ACCESSOR_SUFFIX = {
    PixelFormat.RGBA: "rgba",
    PixelFormat.BGRA: "bgra",
    PixelFormat.RGBA_SRGB: "rgba_srgb",
    PixelFormat.BGRA_SRGB: "bgra_srgb",
    PixelFormat.RGBAF: "rgba_float",
    PixelFormat.RGB: "rgb",
    PixelFormat.RGB_SRGB: "rgb_srgb",
    PixelFormat.BGR: "bgr",
    PixelFormat.BGR_SRGB: "bgr_srgb",
}

_FILL_VIEW_SUFFIX = {
    _zivid.DeviceArrayViewRGBA: "rgba",
    _zivid.DeviceArrayViewBGRA: "bgra",
    _zivid.DeviceArrayViewRGBA_SRGB: "rgba_srgb",
    _zivid.DeviceArrayViewBGRA_SRGB: "bgra_srgb",
    _zivid.DeviceArrayViewRGBAf: "rgba_float",
    _zivid.DeviceArrayViewRGB: "rgb",
    _zivid.DeviceArrayViewRGB_SRGB: "rgb_srgb",
    _zivid.DeviceArrayViewBGR: "bgr",
    _zivid.DeviceArrayViewBGR_SRGB: "bgr_srgb",
}


class Frame2D:
    """A 2D frame captured by a Zivid camera.

    Contains a 2D image as well as metadata, settings and state of the API at the time of capture.

    The images are not corrected for lens distortion. If your application relies on the geometry of the
    image, you can undistort it using the camera intrinsics.
    """

    def __init__(self, impl):
        """Initialize Frame2D wrapper.

        Args:
            impl:   A pathlib.Path or str path to a .zdf file, or a reference to an internal
                    _zivid.Frame2D instance.

        Raises:
            TypeError: If argument does not match the expected types.
        """
        if isinstance(impl, (str, Path)):
            self.__impl = _zivid.Frame2D(str(impl))
        elif isinstance(impl, _zivid.Frame2D):
            self.__impl = impl
        else:
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}, {} or {}.".format(
                    type(impl), str, Path, _zivid.Frame2D
                )
            )

    def __str__(self):
        return str(self.__impl)

    def image_rgba(self) -> Image:
        """Get color (RGBA) image from the frame.

        Returns:
            An image instance containing RGBA data
        """
        return Image(self.__impl.image_rgba())

    def image_bgra(self) -> Image:
        """Get color (BGRA) image from the frame.

        Returns:
            An image instance containing BGRA data
        """
        return Image(self.__impl.image_bgra())

    def image_rgba_srgb(self) -> Image:
        """Get color (RGBA) image from the frame in the sRGB color space.

        Returns:
            An image instance containing RGBA data in sRGB color space
        """
        return Image(self.__impl.image_rgba_srgb())

    def image_bgra_srgb(self) -> Image:
        """Get color (BGRA) image from the frame in the sRGB color space.

        Returns:
            An image instance containing BGRA data in sRGB color space
        """
        return Image(self.__impl.image_bgra_srgb())

    def image_rgb(self) -> Image:
        """Get color (RGB, 3-channel, no alpha) image from the frame.

        Identical to :py:meth:`image_rgba` but skips the alpha channel for ~25% bandwidth savings.
        """
        return Image(self.__impl.image_rgb())

    def image_rgb_srgb(self) -> Image:
        """Get color (RGB, 3-channel, no alpha) image from the frame in the sRGB color space."""
        return Image(self.__impl.image_rgb_srgb())

    def image_bgr(self) -> Image:
        """Get color (BGR, 3-channel, no alpha) image from the frame."""
        return Image(self.__impl.image_bgr())

    def image_bgr_srgb(self) -> Image:
        """Get color (BGR, 3-channel, no alpha) image from the frame in the sRGB color space."""
        return Image(self.__impl.image_bgr_srgb())

    def image_device_array(self, stream_or_queue, color_format: PixelFormat) -> DeviceArray:
        """Get the 2D image as a newly allocated GPU device buffer.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr the SDK records a readiness
                event on before handing off the buffer.
            color_format: A zivid.PixelFormat color format. All color formats are supported: the
                4-channel RGBA / BGRA / RGBA_SRGB / BGRA_SRGB / RGBAF and the 3-channel RGB / BGR /
                RGB_SRGB / BGR_SRGB.

        Returns:
            A DeviceArray owning the freshly allocated buffer.
        """
        suffix = _resolve_color_format(color_format, _COLOR_FORMAT_ACCESSOR_SUFFIX)
        _require_stream_or_queue(stream_or_queue)
        accessor = getattr(self.__impl, "image_device_array_{}".format(suffix))
        return DeviceArray(accessor(stream_or_queue))

    def image_device_array_fill(self, stream_or_queue, destination_buffer: DeviceArrayView) -> None:
        """Fill a caller-provided DeviceArrayView with the 2D image.

        The color format is taken from destination_buffer, which must be a color DeviceArrayView
        created via zivid.create_device_array_view. All color formats can be filled, including the
        3-channel RGB / BGR / RGB_SRGB / BGR_SRGB.

        Args:
            stream_or_queue: A CUDAStreamPtr or OpenCLCommandQueuePtr the SDK enqueues the fill on.
            destination_buffer: A color DeviceArrayView to fill in place.

        Raises:
            TypeError: If destination_buffer is not a color DeviceArrayView.
        """
        suffix, impl = _device_array_view_fill_target(destination_buffer, _FILL_VIEW_SUFFIX)
        _require_stream_or_queue(stream_or_queue)
        fill = getattr(self.__impl, "image_device_array_{}_fill".format(suffix))
        fill(stream_or_queue, impl)

    def image_srgb(self) -> Image:
        """Get color (RGBA) image from the frame in the sRGB color space.

        This method is deprecated. Use image_rgba_srgb() instead.

        Returns:
            An image instance containing RGBA data in sRGB color space
        """
        return Image(self.__impl.image_rgba_srgb())

    def save(self, file_path: str | Path) -> None:
        """Save the 2D frame to a .zdf file.

        Args:
            file_path: A pathlib.Path instance or a string specifying the destination .zdf file
        """
        self.__impl.save(str(file_path))

    def load(self, file_path: str | Path) -> None:
        """Load a 2D frame from a .zdf file.

        Args:
            file_path: A pathlib.Path instance or a string specifying the .zdf file to load
        """
        self.__impl.load(str(file_path))

    @property
    def settings(self) -> Settings2D:
        """Get the settings used to capture this frame.

        Returns:
            A Settings2D instance
        """
        return _to_settings2d(self.__impl.settings)

    @property
    def state(self) -> CameraState:
        """Get the camera state data at the time of the frame capture.

        Returns:
            A CameraState instance
        """
        return _to_camera_state(self.__impl.state)

    @property
    def info(self) -> FrameInfo:
        """Get information collected at the time of the capture.

        Returns:
            A FrameInfo instance
        """
        return _to_frame_info(self.__impl.info)

    @property
    def camera_info(self) -> CameraInfo:
        """Get information about the camera used to capture the frame.

        Returns:
            A CameraInfo instance
        """
        return _to_camera_info(self.__impl.camera_info)

    def release(self) -> None:
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def clone(self) -> Frame2D:
        """Get a clone of the frame.

        The clone will include a copy of all the frame data.

        This function incurs a performance cost due to the copying of the data. When performance is important we
        recommend to avoid using this method, and instead modify the existing frame.

        This method is equivalent to calling `copy.deepcopy` on the frame. You can obtain a shallow copy that does
        not copy the underlying data by using `copy.copy` on the frame instead.

        Returns:
            A Frame2D instance
        """
        return Frame2D(self.__impl.clone())

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()

    def __copy__(self):
        return Frame2D(self.__impl.__copy__())

    def __deepcopy__(self, memodict):
        return Frame2D(self.__impl.__deepcopy__(memodict))
