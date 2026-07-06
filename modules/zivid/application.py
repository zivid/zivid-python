"""Contains Application class."""

import _zivid
from zivid.camera import Camera
from zivid.camera_address import _to_internal_camera_address


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

    def __init__(self, cuda_context=None):
        """Initialize application.

        Args:
            cuda_context: Optional CUDAContextPtr to use a user-provided CUDA context.
                          This ensures Zivid uses the same CUDA context as your application,
                          which is required for GPU interoperability with libraries like
                          PyTorch, CuPy, or other CUDA-based frameworks.

        Example:
                              import cupy as cp
                              # Get CuPy's CUDA context
                              cuda_ctx = cp.cuda.runtime.cudaGetDevice()
                              ctx_ptr = ...  # Get context pointer
                              app = zivid.Application(zivid.CUDAContextPtr(ctx_ptr))
        """
        if cuda_context is not None:
            self.__impl = _zivid.Application(cuda_context)
        else:
            self.__impl = _zivid.Application()

    def __str__(self):
        return str(self.__impl)

    def create_file_camera(self, frame):
        """Create a virtual camera from a captured frame or a .zfc file.

        A file camera is a virtual camera that replays a previously captured frame. It holds the raw sensor
        data from the original capture and reconstructs point clouds and color images from it, so it can be
        used to develop and test capture pipelines without physical camera hardware.

        Because a file camera owns only the raw data from a single capture, the settings it can capture with
        are more restricted than those of a physical camera:

        - Processing settings (filters, color balance, resampling, etc.) may be changed freely.

        - Engine and Sampling (Pixel and Color) must match the values used for the original capture.
          If set to a value different from the original setting, the capture throws.

        - The number of acquisitions must not exceed the number stored in the file camera. Requesting fewer
          acquisitions is allowed and replays the corresponding subset; requesting more throws.

        - Acquisition values (Aperture, ExposureTime, Gain, Brightness) do not affect the captured data.
          The requested settings are recorded on the returned frame, but they do not change the underlying
          sensor data.

        - Settings left unset are set automatically before capture: Engine and Sampling are set from the file
          camera, while unset acquisition and processing fields are filled with the camera-model defaults.

        The same restrictions apply to the 2D color acquisitions of a 2D or 2D+3D capture.

        When called with a Frame, the frame must have been captured with
        Settings.Diagnostics.Enabled.

        When called with a file path, the file must be a Zivid File Camera (.zfc) file.
        This form is deprecated and will be removed in SDK 3.0; use the Frame overload instead.

        An example file camera may be found among the Sample Data at zivid.com/downloads

        Args:
            frame: A Frame captured with diagnostics enabled, or a pathlib.Path / string
                pointing to a .zfc file (deprecated)

        Returns:
            Zivid virtual Camera instance
        """
        from zivid.frame import Frame  # pylint: disable=import-outside-toplevel

        if isinstance(frame, Frame):
            # pylint: disable=protected-access
            return Camera(self.__impl.create_file_camera(frame._Frame__impl))
        return Camera(self.__impl.create_file_camera(str(frame)))

    def connect_camera(self, serial_number=None, address=None):
        """Connect to the next available Zivid camera.

        Args:
            serial_number: Optional serial number string for connecting to a specific camera
            address: Optional CameraAddress for connecting directly by hostname or IPv4 address,
                     bypassing mDNS discovery

        Returns:
            Zivid Camera instance

        Raises:
            ValueError: If both serial_number and address are provided
        """
        if serial_number is not None and address is not None:
            raise ValueError("Cannot specify both serial_number and address")
        if address is not None:
            return Camera(self.__impl.connect_camera(_to_internal_camera_address(address)))
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

    def compute_device(self):
        """Get the GPU compute device used by the Application.

        Returns:
            ComputeDevice instance with information about the GPU being used
        """
        return self.__impl.compute_device()

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
