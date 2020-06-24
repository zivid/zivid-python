import platform

import pytest


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason=r"python: can't open file 'D:\a\samples\sample_capture_from_file.py': [Errno 2] No such file or directory",
)
def test_capture_from_file(sample_data_file):
    pytest.helpers.run_sample(
        name="capture_from_file", working_directory=sample_data_file.parent
    )


@pytest.mark.skipif(
    platform.system() == "Windows",
    reason=r"python: can't open file 'D:\a\samples\sample_print_version_info.py': [Errno 2] No such file or directory",
)
def test_print_version_info():
    pytest.helpers.run_sample(name="print_version_info")
