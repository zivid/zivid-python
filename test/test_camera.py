import pytest


def test_illegal_init(application):
    import zivid

    with pytest.raises(TypeError):
        zivid.camera.Camera("this should fail")

    with pytest.raises(TypeError):
        zivid.camera.Camera(None)

    with pytest.raises(TypeError):
        zivid.camera.Camera(12345)


def test_init_with(application, file_camera_file):
    import zivid

    with application.create_file_camera(file_camera_file) as file_camera:
        assert file_camera
        assert isinstance(file_camera, zivid.camera.Camera)


def test_capture_settings(shared_file_camera):
    import zivid

    frame = shared_file_camera.capture(
        zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    )
    assert frame
    assert isinstance(frame, zivid.frame.Frame)
    frame.release()


def test_capture_settings_2d(shared_file_camera):
    import zivid

    frame_2d = shared_file_camera.capture(
        zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
    )
    assert frame_2d
    assert isinstance(frame_2d, zivid.frame_2d.Frame2D)
    frame_2d.release()


def test_equal(application, file_camera_file):
    import zivid

    with application.create_file_camera(file_camera_file) as file_camera:
        camera_handle = zivid.Camera(
            file_camera._Camera__impl  # pylint: disable=protected-access
        )
        assert isinstance(file_camera, zivid.Camera)
        assert isinstance(camera_handle, zivid.Camera)
        assert camera_handle == file_camera


def test_not_equal(application, file_camera_file):
    with application.create_file_camera(
        file_camera_file
    ) as file_camera1, application.create_file_camera(file_camera_file) as file_camera2:
        assert file_camera1 != file_camera2


def test_disconnect(file_camera):
    assert file_camera.state.connected
    file_camera.disconnect()
    assert not file_camera.state.connected


def test_connect(file_camera):
    file_camera.disconnect()
    assert not file_camera.state.connected
    file_camera.connect()
    assert file_camera.state.connected


def test_connect_capture_chaining(file_camera):
    import zivid

    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    file_camera.disconnect()
    file_camera.connect().capture(settings)


def test_to_string(shared_file_camera):
    string = str(shared_file_camera)
    assert string
    assert isinstance(string, str)


def test_info(shared_file_camera):
    import zivid

    info = shared_file_camera.info
    assert info
    assert isinstance(info, zivid.CameraInfo)


def test_state(shared_file_camera):
    import zivid

    state = shared_file_camera.state
    assert state
    assert isinstance(state, zivid.CameraState)


@pytest.mark.physical_camera
def test_measure_scene_conditions(physical_camera):
    from zivid import SceneConditions

    scene_conditions = physical_camera.measure_scene_conditions()
    assert scene_conditions
    assert isinstance(scene_conditions, SceneConditions)
    assert (
        scene_conditions.ambient_light.flicker_classification
        in SceneConditions.AmbientLight.FlickerClassification.valid_values()
    )


def test_measure_scene_conditions_fails_with_file_camera(file_camera):
    with pytest.raises(
        RuntimeError, match="This camera cannot measure surrounding conditions"
    ):
        file_camera.measure_scene_conditions()
