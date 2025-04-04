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

from zivid.experimental.point_cloud_export.file_format import IncludeNormals


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


def test_point_cloud_export_as_ply_with_normals(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.ply")
        export_frame(frame, PLY(filepath, include_normals=IncludeNormals.yes))
        assert os.path.exists(filepath)

        filepath_no_normals = os.path.join(tmpdir, "frame_no_normals.ply")
        export_frame(frame, PLY(filepath_no_normals, include_normals=IncludeNormals.no))
        assert os.path.exists(filepath_no_normals)

        # ensure that the file with normals is bigger than the one without
        assert os.path.getsize(filepath) > os.path.getsize(filepath_no_normals)


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


def test_point_cloud_export_as_pcd_with_normals(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame.pcd")
        export_frame(frame, PCD(filepath, include_normals=IncludeNormals.yes))
        assert os.path.exists(filepath)

        filepath_no_normals = os.path.join(tmpdir, "frame_no_normals.pcd")
        export_frame(frame, PCD(filepath_no_normals, include_normals=IncludeNormals.no))
        assert os.path.exists(filepath_no_normals)

        # ensure that the file with normals is bigger than the one without
        assert os.path.getsize(filepath) > os.path.getsize(filepath_no_normals)
