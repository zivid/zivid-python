"""Module defining file formats that point cloud data can be exported to."""

import _zivid


class ColorSpace:  # pylint: disable=too-few-public-methods
    """Color space for saving point cloud."""

    linear_rgb = "linear_rgb"
    srgb = "srgb"

    @staticmethod
    def valid_values():
        """Get valid values for color space.

        Returns:
            List of valid color spaces.
        """
        return [ColorSpace.linear_rgb, ColorSpace.srgb]

    @classmethod
    def _to_internal(cls, value):
        if value == ColorSpace.linear_rgb:
            return _zivid.point_cloud_export.ColorSpace.linear_rgb
        if value == ColorSpace.srgb:
            return _zivid.point_cloud_export.ColorSpace.srgb
        raise ValueError(
            "Invalid color space '{}'. Valid color spaces are: {}".format(
                value, cls.valid_values()
            )
        )


class ZDF:  # pylint: disable=too-few-public-methods
    """Specification for saving frame in ZDF (*.zdf) format."""

    def __init__(self, file_name):
        """Create a ZDF file format specification with file name.

        Args:
            file_name: File name.

        Raises:
            TypeError: If file_name is not a string.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {}".format(
                    type(file_name), str
                )
            )
        self.__impl = _zivid.point_cloud_export.file_format.ZDF(file_name)

    def __str__(self):
        return str(self.__impl)


class PLY:  # pylint: disable=too-few-public-methods
    """Specification for saving frame in PLY (*.ply) format.

    PLY is a file format developed at Stanford. To learn more about the PLY file format,
    see https://paulbourke.net/dataformats/ply/.
    """

    class Layout:
        """Layout for saving point cloud."""

        ordered = "ordered"
        unordered = "unordered"

        @staticmethod
        def valid_values():
            """Get valid values for layout.

            Returns:
                List of valid layouts.
            """
            return [PLY.Layout.ordered, PLY.Layout.unordered]

        @classmethod
        def _to_internal(cls, value):
            if value == PLY.Layout.ordered:
                return _zivid.point_cloud_export.file_format.PLY.Layout.ordered
            if value == PLY.Layout.unordered:
                return _zivid.point_cloud_export.file_format.PLY.Layout.unordered
            raise ValueError(
                "Invalid layout '{}'. Valid layouts are: {}".format(
                    value, cls.valid_values()
                )
            )

    def __init__(self, file_name, layout=Layout.ordered, color_space=ColorSpace.srgb):
        """Create a PLY file format specification with file name.

        Args:
            file_name: File name.
            layout: Layout of point cloud. Default is ordered.
            color_space: Color space of point cloud. Default is sRGB.

        Raises:
            TypeError: If file_name, layout, or color_space are not strings.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {}".format(
                    type(file_name), str
                )
            )
        if not isinstance(layout, str):
            raise TypeError(
                "Unsupported type for argument layout. Got {}, expected {}".format(
                    type(layout), str
                )
            )
        if not isinstance(color_space, str):
            raise TypeError(
                "Unsupported type for argument color_space. Got {}, expected {}".format(
                    type(color_space), str
                )
            )
        self.__impl = _zivid.point_cloud_export.file_format.PLY(
            file_name,
            PLY.Layout._to_internal(layout),
            ColorSpace._to_internal(color_space),
        )

    def __str__(self):
        return str(self.__impl)


class XYZ:  # pylint: disable=too-few-public-methods
    """Specification for saving frame in ASCII (*.xyz) format.

    ASCII characters are used to store cartesian coordinates of XYZ points and RGB color values.
    """

    def __init__(self, file_name, color_space=ColorSpace.srgb):
        """Create a XYZ file format specification with file name.

        Sets color space to linear RGB.

        Args:
            file_name: File name.
            color_space: Color space of point cloud. Default is sRGB.

        Raises:
            TypeError: If file_name or color_space are not strings.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {}".format(
                    type(file_name), str
                )
            )
        if not isinstance(color_space, str):
            raise TypeError(
                "Unsupported type for argument color_space. Got {}, expected {}".format(
                    type(color_space), str
                )
            )
        self.__impl = _zivid.point_cloud_export.file_format.XYZ(
            file_name, ColorSpace._to_internal(color_space)
        )

    def __str__(self):
        return str(self.__impl)


class PCD:  # pylint: disable=too-few-public-methods
    """Specification for saving frame in PCD (*.pcd) format.

    PCD is a file format native to the Point Cloud Library (PCL). To learn more about
    the PCD file format, see
    https://pcl.readthedocs.io/projects/tutorials/en/latest/pcd_file_format.html#pcd-file-format.
    """

    def __init__(self, file_name, color_space=ColorSpace.srgb):
        """Create a PCD file format specification with file name.

        Args:
            file_name: File name.
            color_space: Color space of point cloud. Default is sRGB.

        Raises:
            TypeError: If file_name or color_space are not strings.
        """
        if not isinstance(file_name, str):
            raise TypeError(
                "Unsupported type for argument file_name. Got {}, expected {}".format(
                    type(file_name), str
                )
            )
        if not isinstance(color_space, str):
            raise TypeError(
                "Unsupported type for argument color_space. Got {}, expected {}".format(
                    type(color_space), str
                )
            )
        self.__impl = _zivid.point_cloud_export.file_format.PCD(
            file_name,
            ColorSpace._to_internal(color_space),
        )

    def __str__(self):
        return str(self.__impl)
