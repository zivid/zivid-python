import pytest


def test_illegal_init(application):
    import zivid

    with pytest.raises(RuntimeError):
        zivid.frame.Frame("non-exisiting-file.zdf")
    with pytest.raises(TypeError):
        zivid.frame.Frame(None)
    with pytest.raises(TypeError):
        zivid.frame.Frame(12345)


def test_point_cloud(frame):
    import zivid

    point_cloud = frame.point_cloud()
    assert isinstance(point_cloud, zivid.PointCloud)


def test_path_init(application, frame_file):
    from pathlib import Path
    import zivid

    frame = zivid.frame.Frame(frame_file)
    assert isinstance(frame_file, Path)
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


def test_str_as_path_init(application, frame_file):
    import zivid

    frame = zivid.frame.Frame(str(frame_file))
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


def test_save(frame):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "save_test.zdf"
        frame.save(save_path)
        assert save_path.exists()


def test_context_manager(application, frame_file):
    import zivid

    with zivid.frame.Frame(frame_file) as frame:
        frame.point_cloud()
    with pytest.raises(RuntimeError):
        frame.point_cloud()


def test_to_string(frame):
    string = str(frame)
    assert string
    assert isinstance(string, str)


def test_load(frame, frame_file):
    assert frame.load(frame_file) is None


def test_settings(frame):
    from zivid import Settings

    settings = frame.settings
    assert settings
    assert isinstance(settings, Settings)


def test_state(frame):
    from zivid.camera_state import CameraState

    state = frame.state
    assert state
    assert isinstance(state, CameraState)


def test_info(frame):
    from zivid.frame_info import FrameInfo

    info = frame.info
    assert info
    assert isinstance(info, FrameInfo)


def test_camera_info(frame):
    from zivid.camera_info import CameraInfo

    camera_info = frame.camera_info
    assert camera_info
    assert isinstance(camera_info, CameraInfo)


def test_release(frame):
    frame.point_cloud()
    frame.release()
    with pytest.raises(RuntimeError):
        frame.point_cloud()
