import pytest


def test_capture_from_file(file_camera_file):
    pytest.helpers.run_sample(
        name="capture_from_file", working_directory=file_camera_file.parent
    )


def test_print_version_info():
    pytest.helpers.run_sample(name="print_version_info")


@pytest.mark.physical_camera
def test_capture():
    pytest.helpers.run_sample(name="capture")


@pytest.mark.physical_camera
def test_capture_2d():
    pytest.helpers.run_sample(name="capture_2d")


@pytest.mark.physical_camera
def test_capture_assistant():
    pytest.helpers.run_sample(name="capture_assistant")


@pytest.mark.physical_camera
def test_capture_hdr():
    pytest.helpers.run_sample(name="capture_hdr")
