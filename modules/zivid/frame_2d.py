"""Contains the Frame class."""
import _zivid

import zivid._settings_2d_converter as _settings_converter
import zivid._camera_state_converter as _camera_state_converter
import zivid._frame_info_converter as _frame_info_converter
from zivid.image import Image


class Frame2D:
    """A 2D frame captured by a Zivid camera.

    Contains a 2D image as well as metadata, settings and state of the API at the time of capture.
    """

    def __init__(self, internal_frame_2d):
        """Can only be initialized by an internal Zivid 2D frame.

        Args:
            internal_frame_2d: internal 2D frame

        Raises:
            TypeError: unsupported type provided for internal 2d frame

        """
        if isinstance(internal_frame_2d, _zivid.Frame2D):
            self.__impl = internal_frame_2d
        else:
            raise TypeError(
                "Unsupported type for argument internal_frame_2d. Got {}, expected {}.".format(
                    type(internal_frame_2d).__name__, _zivid.Frame2D
                )
            )

    def __str__(self):
        return str(self.__impl)

    def image(self):
        """Return the underlying 2D image.

        Returns:
            an image instance

        """
        return Image(self.__impl.image())

    @property
    def settings(self):
        """Get the settings 2d for the API at the time of the 2d frame capture.

        Returns:
            a settings 2d instance

        """
        return _settings_converter.to_settings_2d(  # pylint: disable=protected-access
            self.__impl.settings
        )

    @property
    def state(self):
        """Get the camera state data at the time of the frame capture.

        Returns:
            a camera state instance

        """
        return _camera_state_converter.to_camera_state(  # pylint: disable=protected-access
            self.__impl.state
        )

    @property
    def info(self):
        """Get information collected at the time of the frame capture.

        Returns:
            a camera info instance

        """
        return _frame_info_converter.to_info(  # pylint: disable=protected-access
            self.__impl.info
        )

    def release(self):
        """Release the underlying resources."""
        self.__impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
