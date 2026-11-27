import tempfile
from pathlib import Path

import zivid


def test_read_frame_file_type_of_frame(frame_file):
    assert zivid.read_frame_file_type(frame_file) == zivid.FrameFileType.frame


def test_read_frame_file_type_accepts_str(frame_file):
    assert zivid.read_frame_file_type(str(frame_file)) == zivid.FrameFileType.frame


def test_read_frame_file_type_of_frame_2d(frame_2d):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "test_frame_2d.zdf"
        frame_2d.save(file_path)
        assert zivid.read_frame_file_type(file_path) == zivid.FrameFileType.frame_2d
