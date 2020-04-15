import _zivid


class DetectionResult:
    def __init__(self, impl):
        self.__impl = impl

    def valid(self):
        return self.__impl.valid()

    def __bool__(self):
        return bool(self.__impl)

    def __str__(self):
        return str(self.__impl)


def detect_feature_points(point_cloud):
    return DetectionResult(
        _zivid.calibration.detect_feature_points(point_cloud._PointCloud__impl)
    )
