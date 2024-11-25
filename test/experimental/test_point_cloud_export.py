import os
import tempfile
from zivid import Frame
from zivid.experimental.point_cloud_export.file_format import (
    ZDF,
    PLY,
    ColorSpace,
    XYZ,
    PCD,
)
from zivid.experimental.point_cloud_export import export_frame


def test_point_cloud_export_as_zdf(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.zdf")
        export_frame(frame, ZDF(filepath))
        assert os.path.exists(filepath)
        loaded_frame = Frame(filepath)
        assert frame.info == loaded_frame.info


def test_point_cloud_export_as_ply_ordered_srgb(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.ply")
        export_frame(
            frame, PLY(filepath, layout=PLY.Layout.ordered, color_space=ColorSpace.srgb)
        )
        assert os.path.exists(filepath)


def test_point_cloud_export_as_ply_unordered_linear(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame_unordered.ply")
        export_frame(
            frame,
            PLY(
                filepath,
                layout=PLY.Layout.unordered,
                color_space=ColorSpace.linear_rgb,
            ),
        )
        assert os.path.exists(filepath)


def test_point_cloud_export_as_xyz_srgb(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.xyz")
        export_frame(frame, XYZ(filepath, color_space=ColorSpace.srgb))
        assert os.path.exists(filepath)


def test_point_cloud_export_as_xyz_linear(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame_linear.xyz")
        export_frame(frame, XYZ(filepath, color_space=ColorSpace.linear_rgb))
        assert os.path.exists(filepath)


def test_point_cloud_export_as_pcd_srgb(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.pcd")
        export_frame(frame, PCD(filepath, color_space=ColorSpace.srgb))
        assert os.path.exists(filepath)


def test_point_cloud_export_as_pcd_linear(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame_linear.pcd")
        export_frame(frame, PCD(filepath, color_space=ColorSpace.linear_rgb))
        assert os.path.exists(filepath)
