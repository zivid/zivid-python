# pylint: disable=missing-module-docstring
import _zivid


class DetectionResult:
    """Class representing detected feature points."""

    def __init__(self, impl):  # noqa: D107
        self.__impl = impl

    def valid(self):
        """Check validity of DetectionResult.

        Returns:
            True if DetectionResult is valid
        """
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def __str__(self):
        return str(self.__impl)


def detect_feature_points(point_cloud):
    """Detect feature points from a calibration object in a point cloud.

    Args:
        point_cloud: PointCloud containing a calibration object

    Returns:
        A DetectionResult instance
    """

    return DetectionResult(
        _zivid.calibration.detect_feature_points(
            point_cloud._PointCloud__impl  # pylint: disable=protected-access
        )
    )
