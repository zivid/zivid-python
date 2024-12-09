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

    def capture_2d_3d(self, settings):
        """Capture a 2D+3D frame.

        This method captures both a 3D point cloud and a 2D color image. Use this method when you want to capture
        colored point clouds. This method will throw if `Settings.color` is not set. Use `capture_3d` for capturing a 3D
        point cloud without a 2D color image.

        These remarks below apply for all capture functions:

        This method returns right after the acquisition of the images is complete, and the camera has stopped projecting
        patterns. Therefore, after this method has returned, the camera can be moved, or objects in the scene can be
        moved, or a capture from another camera with overlapping field of view can be triggered, without affecting the
        point cloud.

        When this method returns, there is still remaining data to transfer from the camera to the PC, and the
        processing of the final point cloud is not completed. Transfer and processing of the point cloud will continue
        in the background. When you call a method on the returned `Frame` object that requires the capture to be
        finished, for example `Frame.point_cloud`, that method will block until the processing is finished and the point
        cloud is available. If an exception occurs after the acquisition of images is complete (during transfer or
        processing of the capture), then that exception is instead thrown when you access the `Frame` object.

        The capture functions can be invoked back-to-back, for doing rapid back-to-back acquisition of multiple (2D or
        3D) captures on the same camera. This is for example useful if you want to do one high-resolution 2D capture
        followed by a lower-resolution 3D capture. The acquisition of the next capture will begin quickly after
        acquisition of the previous capture completed, even when there is remaining transfer and processing for the
        first capture. This allows pipelining several 2D and/or 3D captures, by doing acquisition in parallel with data
        transfer and processing.

        Note: There can be maximum of two in-progress uncompleted 3D (or 2D+3D) captures simultaneously per Zivid
        camera. If you invoke `capture_2d_3d` or `capture_3d` when there are two uncompleted 3D captures in-progress,
        then the capture will not start until the first of the in-progress 3D captures has finished all transfer and
        processing. There is a similar limit of maximum two in-process 2D captures per camera.

        Capture functions can also be called on multiple cameras simultaneously. However, if the cameras have
        overlapping field-of-view then you need to take consideration and sequence the capture calls to avoid the
        captures interfering with each other.

        Args:
            settings: Settings to use for the capture.

        Returns:
            A frame containing a 3D point cloud, a 2D color image, and metadata.

        Raises:
            TypeError: If the settings argument is not a Settings instance.
        """
        if not isinstance(settings, Settings):
            raise TypeError(
                "Unsupported type for argument settings. Got {}, expected {}.".format(
                    type(settings), Settings.__name__
                )
            )
        return Frame(self.__impl.capture_2d_3d(_to_internal_settings(settings)))

    def capture_3d(self, settings):
        """Capture a single 3D frame.

        This method is used to capture a 3D frame without a 2D color image. It ignores all color settings in the input
        settings. See `capture_2d_3d` for capturing a 2D+3D frame.

        This method returns right after the acquisition of the images is complete, and the camera has stopped projecting
        patterns. For more information, see the remarks section of `capture_2d_3d` above. Those remarks apply for both
        2D, 3D, and 2D+3D captures.

        Args:
            settings: Settings to use for the capture.

        Returns:
            A frame containing a 3D point cloud and metadata.

        Raises:
            TypeError: If the settings argument is not a Settings
        """
        if not isinstance(settings, Settings):
            raise TypeError(
                "Unsupported type for argument settings. Got {}, expected {}.".format(
                    type(settings), Settings.__name__
                )
            )
        return Frame(self.__impl.capture_3d(_to_internal_settings(settings)))

    def capture_2d(self, settings):
        """Capture a single 2D frame.

        This method returns right after the acquisition of the images is complete, and the camera has stopped projecting
        patterns. For more information, see the remarks section of `capture_2d_3d` above. Those remarks apply for both
        2D, 3D, and 2D+3D captures.

        Args:
            settings: Settings to use for the capture. Can be either a Settings2D instance or a Settings instance.
                      If a Settings instance is provided, only the Settings.color part is used. An exception is thrown
                      if the Settings.color part is not set.

        Returns:
            A Frame2D containing a 2D image and metadata

        Raises:
            TypeError: If the settings argument is not a Settings2D or a Settings.
        """
        if isinstance(settings, Settings2D):
            return Frame2D(self.__impl.capture_2d(_to_internal_settings2d(settings)))
        if isinstance(settings, Settings):
            return Frame2D(self.__impl.capture_2d(_to_internal_settings(settings)))
        raise TypeError(
            "Unsupported settings type, expected: {expected_types}, got: {value_type}".format(
                expected_types=" or ".join([Settings.__name__, Settings2D.__name__]),
                value_type=type(settings),
            )
        )

    def capture(self, settings):
        """Capture a single frame or a single 2D frame.

        This method is deprecated as of SDK 2.14, and will be removed in the next SDK major version (3.0). Use
        `capture_2d_3d` instead for capturing 2D+3D frames, use `capture_3d` for capturing 3D frames without a 2D color
        image, or use `capture_2d` for capturing a 2D color image only.

        This method shares the common remarks about capture functions as found under `capture_2d_3d`.

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
        raise TypeError(
            "Unsupported settings type, expected: {expected_types}, got: {value_type}".format(
                expected_types=" or ".join([Settings.__name__, Settings2D.__name__]),
                value_type=type(settings),
            )
        )

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
