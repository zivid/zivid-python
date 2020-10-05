"""Contains the Frame class."""
from pathlib import Path
import _zivid

import zivid._settings_converter as _settings_converter
import zivid._camera_state_converter as _camera_state_converter
import zivid._frame_info_converter as _frame_info_converter
from zivid.point_cloud import PointCloud


class Frame:
    """A frame captured by a Zivid camera.

    Contains the point cloud (stored on compute device memory) as well as
    calibration data, settings and state used by the API at time of the frame
    capture. Use point_cloud to access point cloud data.
    """

    def __init__(self, file_name):
        """Create a frame by loading data from a file.

        Args:
            file_name: a pathlib.Path instance or a string

        Raises:
            TypeError: unsupported type provided for file name

        """
        if isinstance(file_name, (str, Path)):
            self.__impl = _zivid.Frame(str(file_name))
        elif isinstance(file_name, _zivid.Frame):
            self.__impl = file_name
        else:
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {} or {}.".format(
                    type(file_name).__name__, str.__name__, Path.__name__
                )
            )

    def __str__(self):
        return str(self.__impl)

    def point_cloud(self):
        """Get the point cloud.

        Returns:
            a point cloud instance

        """
        return PointCloud(self.__impl.point_cloud())

    def save(self, file_path):
        """Save the frame to file. The file type is determined from the file extension.

        Args:
            file_path: destination path

        """
        self.__impl.save(str(file_path))

    def load(self, file_path):
        """Load a frame from a Zivid data file.

        Args:
            file_path: path to zdf file

        """
        self.__impl.load(str(file_path))

    @property
    def settings(self):
        """Get the settings used to capture this frame.

        Returns:
            a settings instance

        """
        return _settings_converter.to_settings(  # pylint: disable=protected-access
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
        return _frame_info_converter.to_frame_info(  # pylint: disable=protected-access
            self.__impl.info
        )

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
