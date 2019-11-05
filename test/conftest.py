import pytest
from scripts.sample_data import download_and_extract


@pytest.fixture(name="application")
def application_fixture():
    import zivid

    with zivid.Application() as app:
        yield app


@pytest.fixture(name="sample_data_file", scope="session")
def sample_data_file_fixture():
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        sample_data = Path(temp_dir) / "MiscObjects.zdf"
        download_and_extract(sample_data)
        yield sample_data


@pytest.fixture(name="file_camera")
def file_camera_fixture(application, sample_data_file):
    with application.create_file_camera(sample_data_file) as file_cam:
        yield file_cam


@pytest.fixture(name="frame")
def frame_fixture(application, sample_data_file):  # pylint: disable=unused-argument
    import zivid

    with zivid.Frame(sample_data_file) as frame:
        yield frame


@pytest.fixture(name="point_cloud")
def point_cloud_fixture(frame):
    with frame.get_point_cloud() as point_cloud:
        yield point_cloud


@pytest.fixture(name="random_settings")
def random_settings_fixture():
    import datetime
    from random import randint, choice, uniform
    import zivid

    heavily_modified_settings = zivid.Settings(
        bidirectional=choice([True, False]),
        blue_balance=uniform(1, 8),
        brightness=uniform(0, 1.8),
        exposure_time=datetime.timedelta(microseconds=randint(6500, 100000)),
        filters=zivid.Settings.Filters(
            contrast=zivid.Settings.Filters.Contrast(
                enabled=choice([True, False]), threshold=uniform(0, 100)
            ),
            outlier=zivid.Settings.Filters.Outlier(
                enabled=choice([True, False]), threshold=uniform(0, 100)
            ),
            saturated=zivid.Settings.Filters.Saturated(enabled=choice([True, False])),
            reflection=zivid.Settings.Filters.Reflection(enabled=choice([True, False])),
            gaussian=zivid.Settings.Filters.Gaussian(
                enabled=choice([True, False]), sigma=uniform(0.5, 5)
            ),
        ),
        gain=uniform(1, 16),
        iris=randint(0, 72),
        red_balance=uniform(1, 8),
    )
    yield heavily_modified_settings


@pytest.fixture(name="three_frames")
def three_frames_fixture(
    application, sample_data_file  # pylint: disable=unused-argument
):
    import zivid

    frames = [zivid.Frame(sample_data_file)] * 3
    yield frames
    for fram in frames:
        fram.release()
