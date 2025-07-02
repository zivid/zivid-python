import tempfile

import pytest


def test_capture_from_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        pytest.helpers.run_sample(name="capture_from_file", working_directory=temp_dir)


def test_print_version_info():
    pytest.helpers.run_sample(name="print_version_info")


@pytest.mark.physical_camera
def test_capture():
    with tempfile.TemporaryDirectory() as temp_dir:
        pytest.helpers.run_sample(name="capture", working_directory=temp_dir)


@pytest.mark.physical_camera
def test_capture_2d():
    with tempfile.TemporaryDirectory() as temp_dir:
        pytest.helpers.run_sample(name="capture_2d", working_directory=temp_dir)


@pytest.mark.physical_camera
def test_capture_assistant():
    with tempfile.TemporaryDirectory() as temp_dir:
        pytest.helpers.run_sample(name="capture_assistant", working_directory=temp_dir)


@pytest.mark.physical_camera
def test_capture_hdr():
    with tempfile.TemporaryDirectory() as temp_dir:
        pytest.helpers.run_sample(name="capture_hdr", working_directory=temp_dir)
