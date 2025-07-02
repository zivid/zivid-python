import os
import tempfile

import pytest
from zivid import Frame
from zivid.experimental.point_cloud_export import export_frame, export_unorganized_point_cloud
from zivid.experimental.point_cloud_export.file_format import PCD, PLY, XYZ, ZDF, ColorSpace, IncludeNormals


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
        export_frame(frame, PLY(filepath, layout=PLY.Layout.ordered, color_space=ColorSpace.srgb))
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


def _read_width_and_height_from_pcd(filepath):
    width = None
    height = None
    with open(filepath, "rb") as f:
        while width is None or height is None:
            line = f.readline()
            if line.startswith(b"WIDTH"):
                width = int(line.split()[1])
            elif line.startswith(b"HEIGHT"):
                height = int(line.split()[1])
    return width, height


def test_point_cloud_export_as_pcd_organized(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame_organized.pcd")
        export_frame(frame, PCD(filepath, layout=PCD.Layout.organized))
        assert os.path.exists(filepath)

        width, height = _read_width_and_height_from_pcd(filepath)
        assert width is not None and height is not None
        assert width == frame.point_cloud().width
        assert height == frame.point_cloud().height


def test_point_cloud_export_as_pcd_unorganized(frame):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "frame_unorganized.pcd")
        export_frame(frame, PCD(filepath, layout=PCD.Layout.unorganized))
        assert os.path.exists(filepath)

        width, height = _read_width_and_height_from_pcd(filepath)
        assert width is not None and height is not None
        assert width == frame.point_cloud().to_unorganized_point_cloud().size
        assert height == 1


def test_export_unorganized_point_cloud_as_ply(frame):
    unorganized_point_cloud = frame.point_cloud().to_unorganized_point_cloud()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "unorganized.ply")
        export_unorganized_point_cloud(unorganized_point_cloud, PLY(filepath, layout=PLY.Layout.unordered))
        assert os.path.exists(filepath)

        with pytest.raises(
            RuntimeError, match="Exporting unorganized point cloud to PLY with ordered layout is not supported"
        ):
            export_unorganized_point_cloud(unorganized_point_cloud, PLY(filepath, layout=PLY.Layout.ordered))


def test_export_unorganized_point_cloud_as_xyz(frame):
    unorganized_point_cloud = frame.point_cloud().to_unorganized_point_cloud()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "unorganized.xyz")
        export_unorganized_point_cloud(unorganized_point_cloud, XYZ(filepath))
        assert os.path.exists(filepath)


def test_export_unorganized_point_cloud_as_pcd(frame):
    unorganized_point_cloud = frame.point_cloud().to_unorganized_point_cloud()
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "unorganized.pcd")
        export_unorganized_point_cloud(unorganized_point_cloud, PCD(filepath, layout=PCD.Layout.unorganized))
        assert os.path.exists(filepath)

        width, height = _read_width_and_height_from_pcd(filepath)
        assert width is not None and height is not None
        assert width == unorganized_point_cloud.size
        assert height == 1

        with pytest.raises(
            RuntimeError, match="Exporting unorganized point cloud to PCD with organized layout is not supported"
        ):
            export_unorganized_point_cloud(unorganized_point_cloud, PCD(filepath, layout=PCD.Layout.organized))
