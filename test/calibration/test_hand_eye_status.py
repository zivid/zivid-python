import _zivid
import zivid


def test_hand_eye_status():
    for value in _zivid.calibration.HandEyeStatus.__members__.values():
        assert getattr(zivid.calibration.HandEyeStatus, value.name) == value.name
