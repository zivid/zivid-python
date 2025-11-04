"""Contains Visualizer class for point cloud visualization."""

import _zivid
from zivid.frame import Frame
from zivid.point_cloud import PointCloud
from zivid.unorganized_point_cloud import UnorganizedPointCloud


class Visualizer:
    """Simple visualizer component for point clouds.

    This is a singleton class that provides a 3D visualization window for displaying
    point clouds, frames, and other 3D data from Zivid cameras.
    """

    def __init__(self):
        """Initialize the Visualizer.

        Creates or gets the singleton visualizer instance.
        """
        self.__impl = _zivid.visualization.Visualizer()

    def __str__(self):
        """Get string representation of the Visualizer.

        Returns:
            String representation of the Visualizer
        """
        return str(self.__impl)

    def show(self, data=None):
        """Show the visualization window or display data.

        When called without arguments, shows the visualization window.
        When called with data, displays the provided point cloud or frame.

        Args:
            data: Optional data to display. Can be a PointCloud, UnorganizedPointCloud, or Frame.
        """
        if data is None:
            self.__impl.show()
        elif isinstance(data, PointCloud):
            self.__impl.show(data._PointCloud__impl)  # pylint: disable=protected-access
        elif isinstance(data, UnorganizedPointCloud):  # pylint: disable=protected-access
            self.__impl.show(data._UnorganizedPointCloud__impl)  # pylint: disable=protected-access
        elif isinstance(data, Frame):
            self.__impl.show(data._Frame__impl)  # pylint: disable=protected-access
        else:
            raise TypeError(f"Unsupported data type for visualization: {type(data)}")

    def hide(self):
        """Hide the visualization window."""
        self.__impl.hide()

    def run(self):
        """Run the event loop.

        Should be called to allow interaction with the point cloud visualization.
        This method will block until the visualization window is closed.

        Returns:
            Exit code from the visualization event loop
        """
        return self.__impl.run()

    def close(self):
        """Stop the event loop and close the window.

        The object goes back to idle state.
        """
        self.__impl.close()

    def resize(self, height, width):
        """Resize the window to specified height and width.

        Args:
            height: Window height in pixels
            width: Window width in pixels
        """
        self.__impl.resize(height, width)

    def reset_to_fit(self):
        """Reset the view so that the point cloud will fit in the window.

        The view will be reset to the default view, which is looking at the point cloud
        along the Z axis in the positive direction.
        """
        self.__impl.reset_to_fit()

    def show_full_screen(self):
        """Show the window in full screen mode."""
        self.__impl.show_full_screen()

    def show_maximized(self):
        """Show the window in maximized mode."""
        self.__impl.show_maximized()

    def set_window_title(self, title):
        """Set the window title.

        Args:
            title: The title string to set for the visualization window
        """
        self.__impl.set_window_title(title)

    @property
    def colors_enabled(self):
        """Whether coloring of the points with their accompanying RGB colors is enabled.

        Returns:
            True if colors are enabled, False otherwise
        """
        return self.__impl.colors_enabled

    @colors_enabled.setter
    def colors_enabled(self, enabled):
        """Enable or disable coloring of the points with their accompanying RGB colors.

        Args:
            enabled: True to enable colors, False to disable
        """
        self.__impl.colors_enabled = enabled

    @property
    def meshing_enabled(self):
        """Whether meshing is enabled.

        Meshing is not supported when showing an unorganized point cloud. An exception
        will be thrown if meshing is enabled while an unorganized point cloud is being shown.

        Returns:
            True if meshing is enabled, False otherwise
        """
        return self.__impl.meshing_enabled

    @meshing_enabled.setter
    def meshing_enabled(self, enabled):
        """Enable or disable meshing.

        Meshing is not supported when showing an unorganized point cloud. An exception
        will be thrown if meshing is enabled while an unorganized point cloud is being shown.

        Args:
            enabled: True to enable meshing, False to disable
        """
        self.__impl.meshing_enabled = enabled

    @property
    def axis_indicator_enabled(self):
        """Whether the axis indicator is enabled.

        Returns:
            True if axis indicator is enabled, False otherwise
        """
        return self.__impl.axis_indicator_enabled

    @axis_indicator_enabled.setter
    def axis_indicator_enabled(self, enabled):
        """Enable or disable the axis indicator.

        Args:
            enabled: True to enable axis indicator, False to disable
        """
        self.__impl.axis_indicator_enabled = enabled

    def release(self):
        """Release the singleton visualizer resources.

        This will invalidate all Visualizer instances and free the underlying resources.
        """
        self.__impl.release()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        """Context manager exit."""
        self.release()
