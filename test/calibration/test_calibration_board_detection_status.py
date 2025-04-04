import _zivid
import zivid


def test_calibration_board_detection_status():
    for (
        value
    ) in _zivid.calibration.CalibrationBoardDetectionStatus.__members__.values():
        assert (
            getattr(zivid.calibration.CalibrationBoardDetectionStatus, value.name)
            == value.name
        )
