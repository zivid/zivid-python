# pylint: disable=import-outside-toplevel


def test_to_internal_camera_state_to_camera_state_modified():
    from zivid import CameraState
    from zivid._camera_state_converter import to_camera_state, to_internal_camera_state

    modified_camera_state = CameraState(connected=True)

    converted_camera_state = to_camera_state(
        to_internal_camera_state(modified_camera_state)
    )
    assert modified_camera_state == converted_camera_state
    assert isinstance(converted_camera_state, CameraState)
    assert isinstance(modified_camera_state, CameraState)


def test_to_internal_camera_state_to_camera_state_default():
    from zivid import CameraState
    from zivid._camera_state_converter import to_camera_state, to_internal_camera_state

    default_camera_state = CameraState()
    converted_camera_state = to_camera_state(
        to_internal_camera_state(default_camera_state)
    )
    assert default_camera_state == converted_camera_state
    assert isinstance(converted_camera_state, CameraState)
    assert isinstance(default_camera_state, CameraState)
