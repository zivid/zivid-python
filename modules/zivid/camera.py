"""Contains Camera class."""
from zivid.frame import Frame
import zivid._settings_converter as _settings_converter
import zivid._camera_state_converter as _camera_state_converter
import _zivid


class Camera:
    """Interface to one Zivid camera."""

    class Revision:
        """Camera revision."""

        def __init__(self, major, minor):
            """Initialize camera revision with major and minor vision.

            Args:
                major: Major hardware revision
                minor: Minor hardware revision

            """
            self.major = major
            self.minor = minor

        def __eq__(self, other):
            return self.major == other.major and self.minor == other.minor

        def __str__(self):
            return "{}.{}".format(self.major, self.minor)

    def __init__(self, internal_camera):
        """Initialize camera with an internal camera.

        Args:
            internal_camera: An internal Zivid camera instance

        """
        if not isinstance(internal_camera, _zivid.Camera):
            raise RuntimeError(
                "Unsupported type for argument internal camera: {}, type: {}.".format(
                    internal_camera, type(internal_camera)
                )
            )
        self.__impl = internal_camera

    def __str__(self):
        return str(self.__impl)

    def __eq__(self, other):
        return self.__impl == other._Camera__impl  # pylint: disable=protected-access

    @property
    def model_name(self):
        """Get the model name.

        Returns:
            A string

        """
        return self.__impl.model_name

    @property
    def revision(self):
        """Get the camera revision.

        Returns:
            A Revision instance

        """
        return Camera.Revision(self.__impl.revision.major, self.__impl.revision.minor)

    @property
    def serial_number(self):
        """Get the serial number of the Zivid camera.

        Returns:
            A string

        """
        return self.__impl.serial_number

    @property
    def firmware_version(self):
        """Get the camera's firmware version.

        Returns:
            A string

        """
        return self.__impl.firmware_version

    def capture(self, settings_collection=None):
        """Capture a single frame.

        Args:
            settings_collection: A collection of settings to be captured and merged into a HDR frame.
                If None, then current settings will be used instead

        Returns:
            A frame containing a 3D image and metadata

        """
        if settings_collection is not None:
            return Frame(
                self.__impl.capture(
                    [
                        _settings_converter.to_internal_settings(settings)
                        for settings in settings_collection
                    ]
                )
            )
        return Frame(self.__impl.capture())

    @property
    def state(self):
        """Get the current camera state.

        Returns:
            The current camera state

        """
        return _camera_state_converter.to_camera_state(self.__impl.state)

    @property
    def settings(self):
        """Get the current camera settings.

        Returns:
            Current settings

        """
        return _settings_converter.to_settings(self.__impl.settings)

    @settings.setter
    def settings(self, settings):
        """Update the camera settings.

        Args:
            settings: New settings for the camera

        """
        self.__impl.settings = _settings_converter.to_internal_settings(  # pylint: disable=protected-access
            settings
        )

    def connect(self, settings=None):
        """Connect to the camera.

        Args:
            settings: New settings for the camera

        """
        if settings is None:
            self.__impl.connect()
        else:
            self.__impl.connect(
                _settings_converter.to_internal_settings(  # pylint: disable=protected-access
                    settings
                )
            )

    def disconnect(self):
        """Disconnect from the camera and free all resources associated with it."""
        self.__impl.disconnect()

    @property
    def user_data_max_size_bytes(self):
        """Return the ammount of data that can be stored in the camera.

        Returns:
            An int

        """
        return self.__impl.user_data_max_size_bytes

    def write_user_data(self, user_data):
        """Write user data to camera. The total number of writes supported depends on camera model and size of data.

        Args:
            user_data: bytes

        """
        if not isinstance(user_data, bytes):
            raise TypeError(
                "Unsupported type for argument user_data. Got {}, expected {}.".format(
                    type(user_data).__name__, bytes.__name__
                )
            )
        self.__impl.write_user_data(list(user_data))

    @property
    def user_data(self):
        """Read user data from camera.

        Returns:
            bytes

        """
        return bytes(self.__impl.user_data)

    def update_settings(self):
        """Return settings updater.

        Use the returned object as a context manager to set settings.

        Returns:
            a settings updater object

        """

        class SettingsUpdater:
            def __init__(self, camera, settings):
                self.__camera = camera
                self.settings = settings

            def __enter__(self):
                return self

            def __exit__(self, exception_type, exception_value, traceback):
                self.__camera.settings = self.settings

        return SettingsUpdater(self, self.settings)

    def release(self):
        """Release the underlying resources."""
        self.__impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
