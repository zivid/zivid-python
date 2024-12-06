"""Contains Camera class."""

import _zivid
from zivid.camera_info import _to_camera_info
from zivid.camera_state import _to_camera_state
from zivid.frame import Frame
from zivid.frame_2d import Frame2D
from zivid.network_configuration import (
    NetworkConfiguration,
    _to_internal_network_configuration,
    _to_network_configuration,
)
from zivid.scene_conditions import _to_scene_conditions
from zivid.settings import Settings, _to_internal_settings
from zivid.settings2d import Settings2D, _to_internal_settings2d


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
                    type(impl), _zivid.Camera
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
        if isinstance(settings, Settings):
            return Frame(self.__impl.capture(_to_internal_settings(settings)))
        if isinstance(settings, Settings2D):
            return Frame2D(self.__impl.capture(_to_internal_settings2d(settings)))
        raise TypeError("Unsupported settings type: {}".format(type(settings)))

    @property
    def info(self):
        """Get information about camera model, serial number etc.

        Returns:
            A CameraInfo instance
        """
        return _to_camera_info(self.__impl.info)

    @property
    def state(self):
        """Get the current camera state.

        Returns:
            A CameraState instance
        """
        return _to_camera_state(self.__impl.state)

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

    @property
    def network_configuration(self):
        """Get the network configuration of the camera.

        Returns:
            NetworkConfiguration instance
        """
        return _to_network_configuration(self.__impl.network_configuration)

    def apply_network_configuration(self, network_configuration):
        """
        Apply the network configuration to the camera.

        Args:
            network_configuration (NetworkConfiguration): The network configuration to apply to the camera.

        This method blocks until the camera has finished applying the network configuration, or raises an exception if
        the camera does not reappear on the network before a timeout occurs.

        This method can be used even if the camera is inaccessible via TCP/IP, for example a camera that is on a
        different subnet to the PC, or a camera with an IP conflict, as it uses UDP multicast to communicate with the
        camera.

        This method can also be used to configure cameras that require a firmware update, as long as the firmware
        supports network configuration via UDP multicast. This has been supported on all firmware versions included
        with SDK 2.10.0 or newer. This method will raise an exception if the camera firmware is too old to support
        UDP multicast.

        This method will raise an exception if the camera status (see CameraState.Status) is "busy", "connected",
        "connecting" or "disconnecting". If the status is "connected", then you must first call disconnect() before
        calling this method.

        Raises:
            TypeError: If the provided network_configuration is not an instance of NetworkConfiguration.
        """
        if not isinstance(network_configuration, NetworkConfiguration):
            raise TypeError(
                "Unsupported type, expected: {expected_type}, got: {value_type}".format(
                    expected_type=NetworkConfiguration,
                    value_type=type(network_configuration),
                )
            )
        self.__impl.apply_network_configuration(
            _to_internal_network_configuration(network_configuration)
        )

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

    def measure_scene_conditions(self):
        """Measure and analyze the conditions of the scene.

        The returned value will report if noticeable ambient light flicker indicative of a 50 Hz or 60 Hz power grid
        was detected. If light flicker is detected in the scene, it is recommended to use capture settings that are
        optimized for that power grid frequency.

        `measure_scene_conditions` will raise a RuntimeException if the camera status (see `CameraState.Status`) is not
        "connected".

        Returns:
            The current scene conditions.
        """
        return _to_scene_conditions(self.__impl.measure_scene_conditions())

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()
