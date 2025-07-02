"""Contains the Frame class."""

import _zivid
from zivid.camera_info import _to_camera_info
from zivid.camera_state import _to_camera_state
from zivid.frame_info import _to_frame_info
from zivid.image import Image
from zivid.settings2d import _to_settings2d


class Frame2D:
    """A 2D frame captured by a Zivid camera.

    Contains a 2D image as well as metadata, settings and state of the API at the time of capture.
    """

    def __init__(self, impl):
        """Initialize Frame2D wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if isinstance(impl, _zivid.Frame2D):
            self.__impl = impl
        else:
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}.".format(type(impl), _zivid.Frame2D)
            )

    def __str__(self):
        return str(self.__impl)

    def image_rgba(self):
        """Get color (RGBA) image from the frame.

        Returns:
            An image instance containing RGBA data
        """
        return Image(self.__impl.image_rgba())

    def image_bgra(self):
        """Get color (BGRA) image from the frame.

        Returns:
            An image instance containing BGRA data
        """
        return Image(self.__impl.image_bgra())

    def image_rgba_srgb(self):
        """Get color (RGBA) image from the frame in the sRGB color space.

        Returns:
            An image instance containing RGBA data in sRGB color space
        """
        return Image(self.__impl.image_rgba_srgb())

    def image_bgra_srgb(self):
        """Get color (BGRA) image from the frame in the sRGB color space.

        Returns:
            An image instance containing BGRA data in sRGB color space
        """
        return Image(self.__impl.image_bgra_srgb())

    def image_srgb(self):
        """Get color (RGBA) image from the frame in the sRGB color space.

        This method is deprecated. Use image_rgba_srgb() instead.

        Returns:
            An image instance containing RGBA data in sRGB color space
        """
        return Image(self.__impl.image_rgba_srgb())

    @property
    def settings(self):
        """Get the settings used to capture this frame.

        Returns:
            A Settings2D instance
        """
        return _to_settings2d(self.__impl.settings)

    @property
    def state(self):
        """Get the camera state data at the time of the frame capture.

        Returns:
            A CameraState instance
        """
        return _to_camera_state(self.__impl.state)

    @property
    def info(self):
        """Get information collected at the time of the capture.

        Returns:
            A FrameInfo instance
        """
        return _to_frame_info(self.__impl.info)

    @property
    def camera_info(self):
        """Get information about the camera used to capture the frame.

        Returns:
            A CameraInfo instance
        """
        return _to_camera_info(self.__impl.camera_info)

    def release(self):
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def clone(self):
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
