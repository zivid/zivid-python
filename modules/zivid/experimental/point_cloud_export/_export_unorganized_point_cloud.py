import _zivid
from zivid.experimental.point_cloud_export.file_format import PCD, PLY, XYZ
from zivid.unorganized_point_cloud import UnorganizedPointCloud


def export_unorganized_point_cloud(unorganized_point_cloud, file_format):
    """Save UnorganizedPointCloud to a file.

    The file format is specified by the file_format argument. The file format can be PLY, XYZ, or
    PCD.

    The layout for PLY file format must be `unordered` and layout for PCD file format must be
    `unorganized`.

    Args:
        unorganized_point_cloud: Unorganized point cloud to export.
        file_format: File format specification.

    Raises:
        TypeError: If unorganized_point_cloud is not a UnorganizedPointCloud.
        TypeError: If file_format is not a file format specification.
    """
    if not isinstance(unorganized_point_cloud, UnorganizedPointCloud):
        raise TypeError(
            "Unsupported type for argument frame. Got {}, expected {}".format(
                type(unorganized_point_cloud), UnorganizedPointCloud
            )
        )
    if not any(
        [
            isinstance(file_format, PLY),
            isinstance(file_format, XYZ),
            isinstance(file_format, PCD),
        ]
    ):
        raise TypeError(
            "Unsupported type for argument file_format. Got {}, expected {}".format(
                type(file_format),
                " or ".join([t.__name__ for t in [PLY, XYZ, PCD]]),
            )
        )

    format_impl_attr = f"_{type(file_format).__name__}__impl"
    _zivid.point_cloud_export.export_unorganized_point_cloud(
        unorganized_point_cloud._UnorganizedPointCloud__impl,  # pylint: disable=protected-access
        getattr(file_format, format_impl_attr),
    )
