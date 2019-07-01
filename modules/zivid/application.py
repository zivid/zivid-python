"""Contains Application class."""
import _zivid
import zivid._settings_converter as _settings_converter
from zivid.camera import Camera


class Application:
    """Manager class for Zivid.

    When the first instance of this class is created it will initialize Zivid
    resources like camera management and GPU managment.

    The resources will exist until release() is called, they will not be
    garbage collected even if the Application instance is. Subsequent instances
    of this class will refer to the already initialized resources.

    Calling release() on one instance of this class will invalidate all other
    instances of the class.

    This class can be used as a context manager to guarantee that resources are
    released deterministically. Note that this will also invalidate all other
    instances of this class.

    """

    def __init__(self):
        """Initialize application."""
        self.__impl = _zivid.Application()

    def __str__(self):
        return str(self.__impl)

    def create_file_camera(self, frame_file, settings=None):
        """Create a virtual camera to simulate Zivid measurements by reading data from a file.

        Args:
            frame_file: Data file in ZDF format containing Zivid data
            settings: Settings for the camera

        Returns:
            Zivid virtual camera instance

        """
        if settings is None:
            return Camera(self.__impl.create_file_camera(str(frame_file)))
        return Camera(
            self.__impl.create_file_camera(
                str(frame_file),
                settings=_settings_converter.to_internal_settings(settings),
            )
        )

    def connect_camera(self, serial_number=None, settings=None):
        """Connect to the next available Zivid Camera.

        Args:
            serial_number: Connect to the camera with this serial number
            settings: Settings for the camera

        Returns:
            Zivid camera instance

        """
        internal_settings = (
            _settings_converter.to_internal_settings(settings) if settings else None
        )
        if serial_number is not None and internal_settings is not None:
            return Camera(
                self.__impl.connect_camera(
                    serial_number=serial_number, settings=internal_settings
                )
            )
        if serial_number is not None:
            return Camera(self.__impl.connect_camera(serial_number))
        if internal_settings is not None:
            return Camera(self.__impl.connect_camera(settings=internal_settings))
        return Camera(self.__impl.connect_camera())

    def cameras(self):
        """Get a list of all cameras.

        Returns:
            A list of cameras including all physical cameras as well as virtual ones
                (e.g. cameras created by create_file_camera())

        """
        return [Camera(internal_camera) for internal_camera in self.__impl.cameras()]

    def release(self):
        """Release the underlying resources."""
        self.__impl.release()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()
