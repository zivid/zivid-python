"""Contains Application class."""
import _zivid
from zivid.camera import Camera


class Application:
    """Manager class for Zivid.

    When the first instance of this class is created it will initialize Zivid
    resources like camera management and GPU management.

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

    def create_file_camera(self, camera_file):
        """Create a virtual camera to simulate Zivid measurements by reading data from a file.

        An example file camera may be found among the Sample Data at zivid.com/downloads

        Args:
            camera_file: A pathlib.Path instance or a string specifying a Zivid File Camera (ZFC) file

        Returns:
            Zivid virtual Camera instance
        """
        return Camera(self.__impl.create_file_camera(str(camera_file)))

    def connect_camera(self, serial_number=None):
        """Connect to the next available Zivid camera.

        Args:
            serial_number: Optional serial number string for connecting to a specific camera

        Returns:
            Zivid Camera instance
        """
        if serial_number is not None:
            return Camera(self.__impl.connect_camera(serial_number))
        return Camera(self.__impl.connect_camera())

    def cameras(self):
        """Get a list of all cameras.

        Returns:
            A list of Camera including all physical cameras as well as virtual ones
                (e.g. cameras created by create_file_camera())
        """
        return [Camera(internal_camera) for internal_camera in self.__impl.cameras()]

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
