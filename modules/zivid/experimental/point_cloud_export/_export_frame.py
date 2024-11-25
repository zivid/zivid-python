from zivid.frame import Frame
from zivid.experimental.point_cloud_export.file_format import ZDF, PLY, XYZ, PCD

import _zivid


def export_frame(frame, file_format):
    """Save frame to a file.

    The file format is specified by the file_format argument. The file format can be ZDF, PLY, XYZ, or PCD.

    If the format is PCD, this function stores the ordered point cloud with a header that indicates an unordered point
    cloud. Since SDK 2.5, it is possible to export PCD with correct header by setting
    `Configuration/APIBreakingBugFixes/FileFormats/PCD/UseOrganizedFormat` in Config.yml file. See
    https://support.zivid.com/en/latest/reference-articles/point-cloud-structure-and-output-formats.html#organized-pcd-format.

    Args:
        frame: Frame to export.
        file_format: File format specification.

    Raises:
        TypeError: If frame is not a Frame.
        TypeError: If file_format is not a file format specification.
    """
    if not isinstance(frame, Frame):
        raise TypeError(
            "Unsupported type for argument frame. Got {}, expected {}".format(
                type(frame), Frame
            )
        )
    if not any(
        [
            isinstance(file_format, ZDF),
            isinstance(file_format, PLY),
            isinstance(file_format, XYZ),
            isinstance(file_format, PCD),
        ]
    ):
        raise TypeError(
            "Unsupported type for argument file_format. Got {}, expected {}".format(
                type(file_format),
                " or ".join([t.__name__ for t in [ZDF, PLY, XYZ, PCD]]),
            )
        )

    format_impl_attr = f"_{type(file_format).__name__}__impl"
    _zivid.point_cloud_export.export_frame(
        frame._Frame__impl,  # pylint: disable=protected-access
        getattr(file_format, format_impl_attr),
    )
