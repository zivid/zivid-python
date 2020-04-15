# pylint: disable=import-outside-toplevel
import pytest


def test_illegal_init(application):  # pylint: disable=unused-argument
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


def test_release(frame):
    frame.point_cloud()
    frame.release()
    with pytest.raises(RuntimeError):
        frame.point_cloud()


def test_path_init(application, sample_point_cloud):  # pylint: disable=unused-argument
    from pathlib import Path
    import zivid

    frame = zivid.frame.Frame(sample_point_cloud)
    assert isinstance(sample_point_cloud, Path)
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


def test_str_as_path_init(
    application, sample_point_cloud  # pylint: disable=unused-argument
):
    import zivid

    frame = zivid.frame.Frame(str(sample_point_cloud))
    assert frame is not None
    assert isinstance(frame, zivid.frame.Frame)


@pytest.mark.skip(reason="https://github.com/zivid/zivid-python/issues/54")
def test_save(frame):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = Path(temp_dir) / "save_test.zdf"
        frame.save(save_path)
        assert save_path.exists()


def test_context_manager(
    application, sample_point_cloud  # pylint: disable=unused-argument
):
    import zivid

    with zivid.frame.Frame(sample_point_cloud) as frame:
        frame.point_cloud()
    with pytest.raises(RuntimeError):
        frame.point_cloud()


def test_to_string(frame):
    string = str(frame)
    assert string
    assert isinstance(string, str)


def test_load(frame, sample_point_cloud):
    assert frame.load(sample_point_cloud) is None


def test_settings(frame):
    with pytest.raises(RuntimeError):
        _ = frame.settings


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
