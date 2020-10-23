"""Contains Camera class."""
from zivid.frame import Frame
from zivid.frame_2d import Frame2D
import zivid
from zivid.settings_2d import Settings2D
import zivid._settings_converter as _settings_converter
import zivid._settings2_d_converter as _settings_2d_converter
import zivid._camera_state_converter as _camera_state_converter
import zivid._camera_info_converter as _camera_info_converter
import _zivid


class Camera:
    """Interface to one Zivid camera.

    This class cannot be initialized directly by the end-user. Use methods on the Application
    class to obtain a Camera instance.
    """

    def __init__(self, impl):
        """Initialize Camera wrapper.

        This constructor is only used internally, and should not be called by the end-user.

        Args:
            impl:   Reference to internal/back-end instance.

        Raises:
            TypeError: If argument does not match the expected internal class.
        """
        if not isinstance(impl, _zivid.Camera):
            raise TypeError(
                "Unsupported type for argument impl. Got {}, expected {}".format(
                    type(impl), type(_zivid.Camera)
                )
            )

        self.__impl = impl

    def __str__(self):
        return str(self.__impl)

    def __eq__(self, other):
        return self.__impl == other._Camera__impl

    def capture(self, settings):
        """Capture a single frame or a single 2D frame.

        Args:
            settings: Settings to be used to capture. Can be either a Settings or Settings2D instance

        Returns:
            A Frame containing a 3D image plus metadata or a Frame2D containing a 2D image plus metadata.

        Raises:
            TypeError: If argument is neither a Settings or a Settings2D
        """
        if isinstance(settings, zivid.Settings):
            return Frame(
                self.__impl.capture(_settings_converter.to_internal_settings(settings))
            )
        if isinstance(settings, Settings2D):
            return Frame2D(
                self.__impl.capture(
                    _settings_2d_converter.to_internal_settings2_d(settings)
                )
            )
        raise TypeError("Unsupported settings type: {}".format(type(settings)))

    @property
    def info(self):
        """Get information about camera model, serial number etc.

        Returns:
            A CameraInfo instance
        """
        return _camera_info_converter.to_camera_info(self.__impl.info)

    @property
    def state(self):
        """Get the current camera state.

        Returns:
            A CameraState instance
        """
        return _camera_state_converter.to_camera_state(self.__impl.state)

    def connect(self):
        """Connect to the camera.

        Returns:
            Reference to the same Camera instance (for chaining)
        """
        self.__impl.connect()
        return self

    def disconnect(self):
        """Disconnect from the camera and free all resources associated with it."""
        self.__impl.disconnect()

    def write_user_data(self, user_data):
        """Write user data to camera. The total number of writes supported depends on camera model and size of data.

        Args:
            user_data: User data as 'bytes' object

        Raises:
            TypeError: Unsupported type provided for user data
        """
        if not isinstance(user_data, bytes):
            raise TypeError(
                "Unsupported type for argument user_data. Got {}, expected {}.".format(
                    type(user_data), bytes.__name__
                )
            )
        self.__impl.write_user_data(list(user_data))

    @property
    def user_data(self):
        """Read user data from camera.

        Returns:
            User data as 'bytes' object
        """
        return bytes(self.__impl.user_data)

    def release(self):
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
