import pytest


def test_capture_from_file(sample_data_file):
    pytest.helpers.run_sample(
        name="capture_from_file", working_directory=sample_data_file.parent
    )


def test_print_version_info():
    pytest.helpers.run_sample(name="print_version_info")
