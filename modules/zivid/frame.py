"""Contains the Frame class."""
from pathlib import Path
import _zivid

import zivid._settings_converter as _settings_converter
import zivid._camera_state_converter as _camera_state_converter
import zivid._camera_info_converter as _camera_info_converter
import zivid._frame_info_converter as _frame_info_converter
from zivid.point_cloud import PointCloud


class Frame:
    """A frame captured by a Zivid camera.

    Contains the point cloud (stored on compute device memory) as well as
    calibration data, settings and state used by the API at time of the frame
    capture. Use the point_cloud() method to access point cloud data.
    """

    def __init__(self, file_name):
        """Create a frame by loading data from a file.

        Args:
            file_name: A pathlib.Path instance or a string specifying a Zivid Data File (ZDF)

        Raises:
            TypeError: Unsupported type provided for file name
        """
        if isinstance(file_name, (str, Path)):
            self.__impl = _zivid.Frame(str(file_name))
        elif isinstance(file_name, _zivid.Frame):
            self.__impl = file_name
        else:
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {} or {}.".format(
                    type(file_name), str.__name__, Path.__name__
                )
            )

    def __str__(self):
        return str(self.__impl)

    def point_cloud(self):
        """Get the point cloud.

        See documentation/functions of zivid.PointCloud for instructions on how to
        retrieve point cloud data on various formats.

        Returns:
            A PointCloud instance
        """
        return PointCloud(self.__impl.point_cloud())

    def save(self, file_path):
        """Save the frame to file. The file type is determined from the file extension.

        Args:
            file_path: A pathlib.Path instance or a string specifying destination

        """
        self.__impl.save(str(file_path))

    def load(self, file_path):
        """Load a frame from a Zivid data file.

        Args:
            file_path: A pathlib.Path instance or a string specifying a ZDF file to load
        """
        self.__impl.load(str(file_path))

    @property
    def settings(self):
        """Get the settings used to capture this frame.

        Returns:
            A Settings instance
        """
        return _settings_converter.to_settings(self.__impl.settings)

    @property
    def state(self):
        """Get the camera state data at the time of the frame capture.

        Returns:
            A CameraState instance
        """
        return _camera_state_converter.to_camera_state(self.__impl.state)

    @property
    def info(self):
        """Get information collected at the time of the frame capture.

        Returns:
            A FrameInfo instance
        """
        return _frame_info_converter.to_frame_info(self.__impl.info)

    @property
    def camera_info(self):
        """Get information about the camera used to capture the frame.

        Returns:
            A CameraInfo instance
        """
        return _camera_info_converter.to_camera_info(self.__impl.camera_info)

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
