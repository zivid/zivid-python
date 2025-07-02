from zivid import CameraState
from zivid.camera_state import _to_camera_state, _to_internal_camera_state


def test_to_internal_camera_state_to_camera_state_modified():
    modified_camera_state = CameraState(connected=True)
    converted_camera_state = _to_camera_state(_to_internal_camera_state(modified_camera_state))
    assert modified_camera_state == converted_camera_state
    assert isinstance(converted_camera_state, CameraState)
    assert isinstance(modified_camera_state, CameraState)


def test_to_internal_camera_state_to_camera_state_default():
    default_camera_state = CameraState()
    converted_camera_state = _to_camera_state(_to_internal_camera_state(default_camera_state))
    assert default_camera_state == converted_camera_state
    assert isinstance(converted_camera_state, CameraState)
    assert isinstance(default_camera_state, CameraState)
