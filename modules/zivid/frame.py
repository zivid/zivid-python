"""Contains the Frame class."""

from pathlib import Path

import _zivid
from zivid.camera_info import _to_camera_info
from zivid.camera_state import _to_camera_state
from zivid.frame_2d import Frame2D
from zivid.frame_info import _to_frame_info
from zivid.point_cloud import PointCloud
from zivid.settings import _to_settings


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

    def frame_2d(self):
        """Get 2D frame from 2D+3D frame.

        If the frame is the result of a 2D+3D capture, this method returns the 2D frame contained in the 2D+3D frame. In
        the case of a 3D-only capture, this method returns None.

        If the frame was captured by an SDK version prior to 2.14.0, then this method will return None.

        If the 2D frame is not yet available because the capture is still in-progress, then this method will block until
        acquisition of the entire 2D+3D capture is done. If you need to access the 2D frame before the 3D acquisition
        has finished, then it is required to do separate 2D and 3D captures.

        In a 2D+3D capture, the 2D color image and 3D point cloud may have different resolutions depending on the pixel
        sampling settings used. The 2D pixel sampling setting determines the resolution of the 2D color image whereas
        the 3D pixel sampling setting and the resampling setting determines the resolution of the 3D point cloud. The 2D
        color image returned in this 2D frame will always have the same resolution as the 2D color image that was
        captured. On the other hand, the point cloud colors will be sampled from the 2D color image to match the
        resolution of the 3D point cloud. The point cloud colors will always have a 1:1 correspondence with the 3D point
        cloud resolution. See `PointCloud` for more information.

        Returns:
            A Frame instance containing the 2D frame, or None if the frame was captured without 2D color or by an SDK
            version prior to 2.14.0.
        """
        return Frame2D(self.__impl.frame_2d()) if self.__impl.frame_2d() is not None else None

    def save(self, file_path):
        """Save the frame to file.

           The file type is determined from the file extension. Supported extensions are .zdf, .ply
           (ordered), .xyz and .pcd.

           If the capture is still in-progress, then this method will block until the capture
           completes.

           This method will copy the necessary point cloud data from the compute device (GPU) memory
           into host memory (RAM), unless that data has already been copied to host memory by an
           earlier function call. This method saves the RGB colors in linear color space for .ply,
           .xyz and .pcd file formats. This method does not save the normal data in the .ply and
           .pcd file formats. The points saved in .ply format will contain NaN (invalid) values. For
           .pcd format, the saved file will contain a header that indicates an unorganized point
           cloud. However the points will contain NaN (invalid) values as well. Since SDK 2.5, it is
           possible to export PCD with correct header (organized) by setting
           Configuration/APIBreakingBugFixes/FileFormats/PCD/UseOrganizedFormat in Config.yml file.
           See
           https://support.zivid.com/en/latest/reference-articles/point-cloud-structure-and-output-formats.html#organized-pcd-format

           It is recommended to use `export_frame` methods to save point cloud to different file
           formats. `export_frame` methods allow customization of the color space of RGB values
           (linear or sRGB), controlling the layout of the points (organized or unorganized) and
           saving normal data in the .ply and .pcd file formats.

           See: zivid.experimental.point_cloud_export(frame, file_format)

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
        return _to_settings(self.__impl.settings)

    @property
    def state(self):
        """Get the camera state data at the time of the frame capture.

        Returns:
            A CameraState instance
        """
        return _to_camera_state(self.__impl.state)

    @property
    def info(self):
        """Get information collected at the time of the frame capture.

        Returns:
            A FrameInfo instance
        """
        return _to_frame_info(self.__impl.info)

    @property
    def camera_info(self):
        """Get information about the camera used to capture the frame.

        Returns:
            A CameraInfo instance
        """
        return _to_camera_info(self.__impl.camera_info)

    def release(self):
        """Release the underlying resources."""
        try:
            impl = self.__impl
        except AttributeError:
            pass
        else:
            impl.release()

    def clone(self):
        """Get a clone of the frame.

        The clone will include a copy of all the point cloud data on the compute device memory. This means that the
        returned frame will not be affected by subsequent modifications on the original frame or point cloud.

        This function incurs a performance cost due to the copying of the compute device memory. When performance is
        important we recommend to avoid using this method, and instead modify the existing frame or point cloud.

        This method is equivalent to calling `copy.deepcopy` on the frame. You can obtain a shallow copy that does
        not copy the underlying data by using `copy.copy` on the frame instead.

        Returns:
            A Frame instance
        """
        return Frame(self.__impl.clone())

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.release()

    def __del__(self):
        self.release()

    def __copy__(self):
        return Frame(self.__impl.__copy__())

    def __deepcopy__(self, memodict):
        return Frame(self.__impl.__deepcopy__(memodict))
