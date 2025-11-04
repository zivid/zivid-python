import copy

import numpy as np
import pytest
import zivid
from assertions import assert_unorganized_point_clouds_equal, assert_unorganized_point_clouds_not_equal


def test_create_empty_unorganized_point_cloud(
    application,  # pylint: disable=unused-argument
):
    upc = zivid.UnorganizedPointCloud()
    assert isinstance(upc, zivid.UnorganizedPointCloud)
    assert upc.size == 0

    xyz = upc.copy_data("xyz")
    xyzw = upc.copy_data("xyzw")
    rgba = upc.copy_data("rgba")
    snr = upc.copy_data("snr")

    assert isinstance(xyz, np.ndarray)
    assert xyz.dtype == np.float32
    assert xyz.shape == (0, 3)

    assert isinstance(xyzw, np.ndarray)
    assert xyzw.dtype == np.float32
    assert xyzw.shape == (0, 4)

    assert isinstance(rgba, np.ndarray)
    assert rgba.dtype == np.uint8
    assert rgba.shape == (0, 4)

    assert isinstance(snr, np.ndarray)
    assert snr.dtype == np.float32
    assert snr.shape == (0,)


def test_create_unorganized_point_cloud(point_cloud):
    n_points_original = point_cloud.width * point_cloud.height
    xyz_organized = point_cloud.copy_data("xyz")
    xyzw_organized = point_cloud.copy_data("xyzw")
    rgba_organized = point_cloud.copy_data("rgba")
    bgra_organized = point_cloud.copy_data("bgra")
    rgba_srgb_organized = point_cloud.copy_data("rgba_srgb")
    bgra_srgb_organized = point_cloud.copy_data("bgra_srgb")
    snr_organized = point_cloud.copy_data("snr")

    valid_mask = ~np.isnan(xyz_organized).any(axis=2)
    n_points_valid = np.sum(valid_mask.flatten())
    assert n_points_valid < n_points_original

    xyz_unorganized_reference = xyz_organized[valid_mask]
    xyzw_unorganized_reference = xyzw_organized[valid_mask]
    rgba_unorganized_reference = rgba_organized[valid_mask]
    bgra_unorganized_reference = bgra_organized[valid_mask]
    rgba_srgb_unorganized_reference = rgba_srgb_organized[valid_mask]
    bgra_srgb_unorganized_reference = bgra_srgb_organized[valid_mask]
    snr_unorganized_reference = snr_organized[valid_mask]

    upc = point_cloud.to_unorganized_point_cloud()
    assert isinstance(upc, zivid.UnorganizedPointCloud)
    assert upc.size == n_points_valid

    xyz = upc.copy_data("xyz")
    assert isinstance(xyz, np.ndarray)
    assert xyz.dtype == np.float32
    assert xyz.shape == (n_points_valid, 3)
    np.testing.assert_array_equal(xyz_unorganized_reference, xyz)

    xyzw = upc.copy_data("xyzw")
    assert isinstance(xyzw, np.ndarray)
    assert xyzw.dtype == np.float32
    assert xyzw.shape == (n_points_valid, 4)
    np.testing.assert_array_equal(xyzw_unorganized_reference, xyzw)

    rgba = upc.copy_data("rgba")
    assert isinstance(rgba, np.ndarray)
    assert rgba.dtype == np.uint8
    assert rgba.shape == (n_points_valid, 4)
    np.testing.assert_array_equal(rgba_unorganized_reference, rgba)

    bgra = upc.copy_data("bgra")
    assert isinstance(bgra, np.ndarray)
    assert bgra.dtype == np.uint8
    assert bgra.shape == (n_points_valid, 4)
    np.testing.assert_array_equal(bgra_unorganized_reference, bgra)

    rgba_srgb = upc.copy_data("rgba_srgb")
    assert isinstance(rgba_srgb, np.ndarray)
    assert rgba_srgb.dtype == np.uint8
    assert rgba_srgb.shape == (n_points_valid, 4)
    np.testing.assert_array_equal(rgba_srgb_unorganized_reference, rgba_srgb)

    bgra_srgb = upc.copy_data("bgra_srgb")
    assert isinstance(bgra_srgb, np.ndarray)
    assert bgra_srgb.dtype == np.uint8
    assert bgra_srgb.shape == (n_points_valid, 4)
    np.testing.assert_array_equal(bgra_srgb_unorganized_reference, bgra_srgb)

    snr = upc.copy_data("snr")
    assert isinstance(snr, np.ndarray)
    assert snr.dtype == np.float32
    assert snr.shape == (n_points_valid,)
    np.testing.assert_array_equal(snr_unorganized_reference, snr)


def test_unorganized_point_cloud_extended(point_cloud):
    upc1 = point_cloud.to_unorganized_point_cloud()
    upc2 = point_cloud.to_unorganized_point_cloud()

    upc_extended = upc1.extended(upc2)
    assert upc_extended.size == 2 * upc1.size

    xyz = upc1.copy_data("xyz")
    xyz_extended = upc_extended.copy_data("xyz")
    assert xyz_extended.shape == (2 * upc1.size, 3)
    np.testing.assert_array_equal(xyz_extended[: upc1.size], xyz)
    np.testing.assert_array_equal(xyz_extended[upc1.size :], xyz)


def test_unorganized_point_cloud_extend_empty(point_cloud):
    upc_canvas = zivid.UnorganizedPointCloud()
    assert upc_canvas.size == 0

    upc = point_cloud.to_unorganized_point_cloud()
    upc_size = upc.size
    assert upc_size > 0

    upc_canvas.extend(upc)
    assert upc.size == upc_size
    assert upc_canvas.size == upc_size

    upc_canvas.extend(upc)
    assert upc.size == upc_size
    assert upc_canvas.size == 2 * upc_size


def test_unorganized_point_cloud_voxel_downsampled(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    upc_downsampled = upc.voxel_downsampled(voxel_size=5.0, min_points_per_voxel=1)
    assert upc_downsampled.size < upc.size


def test_unorganized_point_cloud_transform(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    transform_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    xyz_before = upc.copy_data("xyz")
    upc.transform(transform_matrix)
    xyz_after = upc.copy_data("xyz")
    np.testing.assert_allclose(xyz_after[:, :3], xyz_before[:, :3] + np.array([1.0, 2.0, 3.0]))


def test_unorganized_point_cloud_transformed(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    transform_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    # Get original points
    xyz_before = upc.copy_data("xyz")

    # Get transformed copy
    upc_transformed = upc.transformed(transform_matrix)
    assert isinstance(upc_transformed, zivid.UnorganizedPointCloud)
    assert upc_transformed is not upc

    # Original point cloud should not be modified
    np.testing.assert_array_equal(xyz_before, upc.copy_data("xyz"))

    # New point cloud should have the transformation applied
    xyz_after = upc_transformed.copy_data("xyz")
    np.testing.assert_allclose(xyz_after[:, :3], xyz_before[:, :3] + np.array([1.0, 2.0, 3.0]))


def test_unorganized_point_cloud_center_and_centroid(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud().voxel_downsampled(0.5, 1)
    assert upc.size > 1_000_000, "Must test on large point clouds to check for float precision bugs"

    centroid_before = upc.centroid()
    assert isinstance(centroid_before, np.ndarray)
    xyz_before = upc.copy_data("xyz")
    centroid_before_reference = np.mean(xyz_before, axis=0, dtype=np.float64)

    np.testing.assert_allclose(centroid_before, centroid_before_reference, rtol=1e-4)

    upc.center()
    centroid_after = upc.centroid()
    xyz_after = upc.copy_data("xyz")
    centroid_after_reference = np.mean(xyz_after, axis=0, dtype=np.float64)

    np.testing.assert_allclose(centroid_after, [0.0, 0.0, 0.0], atol=5e-3)
    np.testing.assert_allclose(centroid_after_reference, [0.0, 0.0, 0.0], atol=5e-3)


def test_unorganized_point_cloud_center_and_centroid_empty(point_cloud):
    upc_empty = point_cloud.to_unorganized_point_cloud().voxel_downsampled(1.0, 10000)
    assert upc_empty.size == 0

    centroid = upc_empty.centroid()
    assert centroid is None

    with pytest.raises(RuntimeError):
        upc_empty.center()


def test_illegal_init(
    application,  # pylint: disable=unused-argument
):
    with pytest.raises(TypeError):
        zivid.UnorganizedPointCloud("Should fail.")

    with pytest.raises(TypeError):
        zivid.UnorganizedPointCloud(123)


def test_copy_unorganized_point_cloud(point_cloud, transform):
    upc = point_cloud.to_unorganized_point_cloud()

    with copy.copy(upc) as upc_copy:
        assert isinstance(upc_copy, zivid.UnorganizedPointCloud)
        assert upc_copy is not upc
        assert_unorganized_point_clouds_equal(upc, upc_copy)

        upc.transform(transform)
        # shallow copy, transform should have affected both copies
        assert_unorganized_point_clouds_equal(upc, upc_copy)


def test_deepcopy_unorganized_point_cloud(point_cloud, transform):
    upc = point_cloud.to_unorganized_point_cloud()

    with copy.deepcopy(upc) as upc_copy:
        assert isinstance(upc_copy, zivid.UnorganizedPointCloud)
        assert upc_copy is not upc
        assert_unorganized_point_clouds_equal(upc, upc_copy)

        upc.transform(transform)
        # deep copy, transform should not have affected the copy
        assert_unorganized_point_clouds_not_equal(upc, upc_copy)


def test_clone_point_cloud(point_cloud, transform):
    upc = point_cloud.to_unorganized_point_cloud()

    with upc.clone() as upc_clone:
        assert isinstance(upc_clone, zivid.UnorganizedPointCloud)
        assert upc_clone is not upc
        assert_unorganized_point_clouds_equal(upc, upc_clone)

        upc.transform(transform)
        # clone, transform should not have affected the clone
        assert_unorganized_point_clouds_not_equal(upc, upc_clone)


def test_paint_uniform_color(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    color = [255, 3, 100, 200]
    upc.paint_uniform_color(color)

    rgba = upc.copy_data("rgba")
    np.testing.assert_array_equal(rgba, np.tile(color, (len(rgba), 1)))

    # Numpy array should also be valid input
    upc.paint_uniform_color(np.array([255, 3, 100, 200], dtype=np.uint8))
    upc.paint_uniform_color(np.array([[255, 3, 100, 200]], dtype=np.uint8))


def test_painted_uniform_color(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    color = [255, 3, 100, 200]
    upc_painted = upc.painted_uniform_color(color)

    assert isinstance(upc_painted, zivid.UnorganizedPointCloud)
    assert upc_painted is not upc

    rgba = upc_painted.copy_data("rgba")
    np.testing.assert_array_equal(rgba, np.tile(color, (len(rgba), 1)))

    # Numpy array should also be valid input
    upc.painted_uniform_color(np.array([255, 3, 100, 200], dtype=np.uint8))
    upc.painted_uniform_color(np.array([[255, 3, 100, 200]], dtype=np.uint8))


def test_paint_uniform_color_invalid_color(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    with pytest.raises(RuntimeError):
        upc.paint_uniform_color("Should fail")

    with pytest.raises(RuntimeError):
        upc.paint_uniform_color({255, 1, 2, 100})

    with pytest.raises(RuntimeError):
        upc.paint_uniform_color([255, 0, 0])

    with pytest.raises(RuntimeError):
        upc.paint_uniform_color([255, 0, 0, 1000])

    with pytest.raises(RuntimeError):
        upc.paint_uniform_color([255, 0, 0, 100.2])

    with pytest.raises(TypeError):
        upc.paint_uniform_color(np.array([255, 0, 0, 100]))

    with pytest.raises(TypeError):
        upc.paint_uniform_color(np.array([255, 0, 0, 100], dtype=np.int32))

    with pytest.raises(TypeError):
        upc.paint_uniform_color(np.array([255, 0, 0, 100.2], dtype=np.float16))


def test_painted_uniform_color_invalid_color(point_cloud):
    upc = point_cloud.to_unorganized_point_cloud()

    with pytest.raises(RuntimeError):
        upc.painted_uniform_color("Should fail")

    with pytest.raises(RuntimeError):
        upc.painted_uniform_color({255, 1, 2, 100})

    with pytest.raises(RuntimeError):
        upc.painted_uniform_color([255, 0, 0])

    with pytest.raises(RuntimeError):
        upc.painted_uniform_color([255, 0, 0, 1000])

    with pytest.raises(RuntimeError):
        upc.painted_uniform_color([255, 0, 0, 100.2])

    with pytest.raises(TypeError):
        upc.painted_uniform_color(np.array([255, 0, 0, 100]))

    with pytest.raises(TypeError):
        upc.painted_uniform_color(np.array([255, 0, 0, 100], dtype=np.int32))

    with pytest.raises(TypeError):
        upc.painted_uniform_color(np.array([255, 0, 0, 100.2], dtype=np.float16))
