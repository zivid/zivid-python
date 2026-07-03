import copy

import numpy as np
import pytest
import zivid
from assertions import assert_point_clouds_equal, assert_point_clouds_not_equal


def test_point_cloud_copy_data(point_cloud):
    # Copy all possible formats
    xyz = point_cloud.copy_data("xyz")
    xyzw = point_cloud.copy_data("xyzw")
    xyzrgba = point_cloud.copy_data("xyzrgba")
    xyzbgra = point_cloud.copy_data("xyzbgra")
    rgba = point_cloud.copy_data("rgba")
    bgra = point_cloud.copy_data("bgra")
    srgb = point_cloud.copy_data("srgb")
    normals = point_cloud.copy_data("normals")
    depth = point_cloud.copy_data("z")
    snr = point_cloud.copy_data("snr")
    assert isinstance(xyz, np.ndarray)
    assert isinstance(xyzw, np.ndarray)
    assert isinstance(xyzrgba, np.ndarray)
    assert isinstance(xyzbgra, np.ndarray)
    assert isinstance(rgba, np.ndarray)
    assert isinstance(bgra, np.ndarray)
    assert isinstance(srgb, np.ndarray)
    assert isinstance(normals, np.ndarray)
    assert isinstance(depth, np.ndarray)
    assert isinstance(snr, np.ndarray)

    # Check errors when argument is wrong or missing
    with pytest.raises(ValueError):
        point_cloud.copy_data("bogus-format")
    with pytest.raises(TypeError):
        point_cloud.copy_data()


def test_point_cloud_xyzw(point_cloud):
    xyz = point_cloud.copy_data("xyz")
    xyzw = point_cloud.copy_data("xyzw")
    xyzrgba = point_cloud.copy_data("xyzrgba")
    xyzbgra = point_cloud.copy_data("xyzbgra")
    xyzrgba_srgb = point_cloud.copy_data("xyzrgba_srgb")
    xyzbgra_srgb = point_cloud.copy_data("xyzbgra_srgb")
    depth = point_cloud.copy_data("z")

    assert depth.shape == (point_cloud.height, point_cloud.width)
    assert xyz.shape == (point_cloud.height, point_cloud.width, 3)
    assert xyzw.shape == (point_cloud.height, point_cloud.width, 4)
    assert depth.dtype == np.float32
    assert xyz.dtype == np.float32
    assert xyzw.dtype == np.float32

    for v in "xyz":
        assert xyzrgba[v].dtype == np.float32
        assert xyzbgra[v].dtype == np.float32
        assert xyzrgba_srgb[v].dtype == np.float32
        assert xyzbgra_srgb[v].dtype == np.float32

    np.testing.assert_array_equal(xyz[:, :, 2], depth)
    np.testing.assert_array_equal(xyz[:, :, 0], xyzw[:, :, 0])
    np.testing.assert_array_equal(xyz[:, :, 1], xyzw[:, :, 1])
    np.testing.assert_array_equal(xyz[:, :, 2], xyzw[:, :, 2])


def test_point_cloud_rgba(point_cloud):
    xyzrgba = point_cloud.copy_data("xyzrgba")
    xyzbgra = point_cloud.copy_data("xyzbgra")
    xyzrgba_srgb = point_cloud.copy_data("xyzbgra_srgb")
    xyzbgra_srgb = point_cloud.copy_data("xyzrgba_srgb")
    rgba = point_cloud.copy_data("rgba")
    bgra = point_cloud.copy_data("bgra")
    rgba_srgb = point_cloud.copy_data("rgba_srgb")
    bgra_srgb = point_cloud.copy_data("bgra_srgb")
    srgb = point_cloud.copy_data("srgb")

    assert rgba.shape == (point_cloud.height, point_cloud.width, 4)
    assert bgra.shape == (point_cloud.height, point_cloud.width, 4)
    assert rgba_srgb.shape == (point_cloud.height, point_cloud.width, 4)
    assert bgra_srgb.shape == (point_cloud.height, point_cloud.width, 4)
    assert srgb.shape == (point_cloud.height, point_cloud.width, 4)
    assert xyzrgba.shape == (point_cloud.height, point_cloud.width)
    assert xyzbgra.shape == (point_cloud.height, point_cloud.width)
    assert xyzbgra_srgb.shape == (point_cloud.height, point_cloud.width)
    assert xyzrgba_srgb.shape == (point_cloud.height, point_cloud.width)

    assert rgba.dtype == np.uint8
    assert bgra.dtype == np.uint8
    assert rgba_srgb.dtype == np.uint8
    assert bgra_srgb.dtype == np.uint8
    assert srgb.dtype == np.uint8

    for v in "rgba":
        assert xyzrgba[v].dtype == np.uint8
        assert xyzbgra[v].dtype == np.uint8
        assert xyzrgba_srgb[v].dtype == np.uint8
        assert xyzbgra_srgb[v].dtype == np.uint8

    # rgba equal to xyzrgba colors
    np.testing.assert_array_equal(rgba[:, :, 0], xyzrgba["r"])
    np.testing.assert_array_equal(rgba[:, :, 1], xyzrgba["g"])
    np.testing.assert_array_equal(rgba[:, :, 2], xyzrgba["b"])
    np.testing.assert_array_equal(rgba[:, :, 3], xyzrgba["a"])

    # bgra equal to xyzbgra colors
    np.testing.assert_array_equal(bgra[:, :, 0], xyzbgra["b"])
    np.testing.assert_array_equal(bgra[:, :, 1], xyzbgra["g"])
    np.testing.assert_array_equal(bgra[:, :, 2], xyzbgra["r"])
    np.testing.assert_array_equal(bgra[:, :, 3], xyzbgra["a"])

    # xyzbgra color equal to xyzrgba colors
    np.testing.assert_array_equal(xyzbgra["r"], xyzrgba["r"])
    np.testing.assert_array_equal(xyzbgra["g"], xyzrgba["g"])
    np.testing.assert_array_equal(xyzbgra["b"], xyzrgba["b"])
    np.testing.assert_array_equal(xyzbgra["a"], xyzrgba["a"])

    # bgra equal to rgba
    np.testing.assert_array_equal(bgra[:, :, 0], rgba[:, :, 2])
    np.testing.assert_array_equal(bgra[:, :, 1], rgba[:, :, 1])
    np.testing.assert_array_equal(bgra[:, :, 2], rgba[:, :, 0])
    np.testing.assert_array_equal(bgra[:, :, 3], rgba[:, :, 3])

    # srgb equal to xyzrgba_srgb
    np.testing.assert_array_equal(srgb[:, :, 0], xyzrgba_srgb["r"])
    np.testing.assert_array_equal(srgb[:, :, 1], xyzrgba_srgb["g"])
    np.testing.assert_array_equal(srgb[:, :, 2], xyzrgba_srgb["b"])
    np.testing.assert_array_equal(srgb[:, :, 3], xyzrgba_srgb["a"])

    # bgra_srgb equal to xyzbgra_srgb
    np.testing.assert_array_equal(bgra_srgb[:, :, 0], xyzbgra_srgb["b"])
    np.testing.assert_array_equal(bgra_srgb[:, :, 1], xyzbgra_srgb["g"])
    np.testing.assert_array_equal(bgra_srgb[:, :, 2], xyzbgra_srgb["r"])
    np.testing.assert_array_equal(bgra_srgb[:, :, 3], xyzbgra_srgb["a"])

    # xyzbgra_srgb color equal to xyzrgba_srgb colors
    np.testing.assert_array_equal(xyzbgra_srgb["r"], xyzrgba_srgb["r"])
    np.testing.assert_array_equal(xyzbgra_srgb["g"], xyzrgba_srgb["g"])
    np.testing.assert_array_equal(xyzbgra_srgb["b"], xyzrgba_srgb["b"])
    np.testing.assert_array_equal(xyzbgra_srgb["a"], xyzrgba_srgb["a"])

    # bgra_srgb equal to srgb
    np.testing.assert_array_equal(bgra_srgb[:, :, 0], srgb[:, :, 2])
    np.testing.assert_array_equal(bgra_srgb[:, :, 1], srgb[:, :, 1])
    np.testing.assert_array_equal(bgra_srgb[:, :, 2], srgb[:, :, 0])
    np.testing.assert_array_equal(bgra_srgb[:, :, 3], srgb[:, :, 3])

    # rgba_srgb equal to srgb
    np.testing.assert_array_equal(rgba_srgb[:, :, 0], srgb[:, :, 0])
    np.testing.assert_array_equal(rgba_srgb[:, :, 1], srgb[:, :, 1])
    np.testing.assert_array_equal(rgba_srgb[:, :, 2], srgb[:, :, 2])
    np.testing.assert_array_equal(rgba_srgb[:, :, 3], srgb[:, :, 3])

    # rgba_srgb equal to xyzrgba_srgb
    np.testing.assert_array_equal(rgba_srgb[:, :, 0], xyzrgba_srgb["r"])
    np.testing.assert_array_equal(rgba_srgb[:, :, 1], xyzrgba_srgb["g"])
    np.testing.assert_array_equal(rgba_srgb[:, :, 2], xyzrgba_srgb["b"])
    np.testing.assert_array_equal(rgba_srgb[:, :, 3], xyzrgba_srgb["a"])


def test_point_cloud_copy_image(point_cloud):
    image_rgba = point_cloud.copy_image("rgba")
    assert isinstance(image_rgba, zivid.Image)
    assert image_rgba.height == point_cloud.height
    assert image_rgba.width == point_cloud.width

    image_bgra = point_cloud.copy_image("bgra")
    assert isinstance(image_bgra, zivid.Image)
    assert image_bgra.height == point_cloud.height
    assert image_bgra.width == point_cloud.width

    image_srgb = point_cloud.copy_image("srgb")
    assert isinstance(image_srgb, zivid.Image)
    assert image_srgb.height == point_cloud.height
    assert image_srgb.width == point_cloud.width

    rgba = point_cloud.copy_data("rgba")
    np.testing.assert_array_equal(image_rgba.copy_data(), rgba)

    bgra = point_cloud.copy_data("bgra")
    np.testing.assert_array_equal(image_bgra.copy_data(), bgra)

    srgb = point_cloud.copy_data("srgb")
    np.testing.assert_array_equal(image_srgb.copy_data(), srgb)

    # Check errors when argument is wrong or missing
    with pytest.raises(ValueError):
        point_cloud.copy_image("bogus-format")
    with pytest.raises(TypeError):
        point_cloud.copy_image()


def test_point_cloud_normals(point_cloud):
    normals = point_cloud.copy_data("normals")

    assert normals.shape == (point_cloud.height, point_cloud.width, 3)
    assert normals.dtype == np.float32

    normals_flat = normals.reshape(normals.shape[0] * normals.shape[1], 3)
    non_nan_normals = normals_flat[~np.isnan(normals_flat[:, 0])]
    vector_lengths = np.linalg.norm(non_nan_normals, axis=1)
    np.testing.assert_array_almost_equal(vector_lengths, 1.0)


def test_point_cloud_snr(point_cloud):
    snr = point_cloud.copy_data("snr")

    assert snr.dtype == np.float32
    assert snr.shape == (point_cloud.height, point_cloud.width)
    assert np.all(snr >= 0.0)
    assert np.all(snr < 1000)


def test_height(point_cloud):
    height = point_cloud.height

    assert height is not None
    assert isinstance(height, int)


def test_width(point_cloud):
    width = point_cloud.width

    assert width is not None
    assert isinstance(width, int)


def _validate_transformation(xyzw_before, xyzw_after, transform):
    # Pick an arbitary point to test
    i = xyzw_before.shape[0] // 3
    j = xyzw_before.shape[1] // 3
    point_before = xyzw_before[i, j, :]
    point_after = xyzw_after[i, j, :]
    assert np.all(~np.isnan(point_after))
    assert np.all(~np.isnan(point_before))
    point_after_expected = np.dot(transform, point_before)
    np.testing.assert_allclose(point_after, point_after_expected, rtol=1e-6)


def test_transform(point_cloud, transform):
    # Get points before and after transform
    xyzw_before = point_cloud.copy_data("xyzw")
    point_cloud_returned = point_cloud.transform(transform)
    xyzw_after = point_cloud.copy_data("xyzw")

    np.testing.assert_array_equal(point_cloud_returned.transformation_matrix, transform)

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the transformation was actually applied
    _validate_transformation(xyzw_before, xyzw_after, transform)


def test_transformed(point_cloud, transform):
    # Get original points
    xyzw_before = point_cloud.copy_data("xyzw")

    # Get transformed copy
    point_cloud_transformed = point_cloud.transformed(transform)
    assert isinstance(point_cloud_transformed, zivid.PointCloud)
    assert point_cloud_transformed is not point_cloud

    # Original point cloud should not be modified
    np.testing.assert_array_equal(xyzw_before, point_cloud.copy_data("xyzw"))

    # New point cloud should have the transformation applied
    xyzw_after = point_cloud_transformed.copy_data("xyzw")
    _validate_transformation(xyzw_before, xyzw_after, transform)


def test_transform_chaining(point_cloud, transform):
    # Get points before and after transform
    xyzw_before = point_cloud.copy_data("xyzw")
    point_cloud_returned = point_cloud.transform(transform).transform(transform)
    xyzw_after = point_cloud.copy_data("xyzw")

    chained_transform = np.dot(transform, transform)
    np.testing.assert_array_equal(point_cloud_returned.transformation_matrix, chained_transform)

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the transformation was actually applied
    _validate_transformation(xyzw_before, xyzw_after, chained_transform)


def test_downsampling_enum():
    vals = zivid.PointCloud.Downsampling.valid_values()
    assert len(vals) == 3
    assert "by2x2" in vals
    assert "by3x3" in vals
    assert "by4x4" in vals
    assert zivid.PointCloud.Downsampling.by2x2 == "by2x2"
    assert zivid.PointCloud.Downsampling.by3x3 == "by3x3"
    assert zivid.PointCloud.Downsampling.by4x4 == "by4x4"


def _make_downsampling_enum(fraction):
    return "by{f}x{f}".format(f=fraction)


@pytest.mark.parametrize("fraction", [2, 3, 4])
def test_downsample(point_cloud, fraction):
    # Remember original size
    height_orig = point_cloud.height
    width_orig = point_cloud.width

    # Perform downsampling
    point_cloud_returned = point_cloud.downsample(_make_downsampling_enum(fraction))

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the new size is as expected
    assert height_orig // fraction == point_cloud.height
    assert width_orig // fraction == point_cloud.width


@pytest.mark.parametrize("fraction", [2, 3, 4])
def test_downsampled(point_cloud, fraction):
    # Remember original size
    height_orig = point_cloud.height
    width_orig = point_cloud.width

    # Perform downsampling
    point_cloud_new = point_cloud.downsampled(_make_downsampling_enum(fraction))

    # Check that a new object was returned and that the original is untouched
    assert isinstance(point_cloud_new, zivid.PointCloud)
    assert point_cloud_new is not point_cloud
    assert point_cloud.height == height_orig
    assert point_cloud.width == width_orig

    # Check that the new size is as expected
    assert height_orig // fraction == point_cloud_new.height
    assert width_orig // fraction == point_cloud_new.width


def test_downsample_chaining(point_cloud):
    # Remember original size
    height_orig = point_cloud.height
    width_orig = point_cloud.width

    # Perform downsampling
    point_cloud_returned = point_cloud.downsample("by2x2").downsample("by3x3")

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the new size is as expected
    assert height_orig // 2 // 3 == point_cloud.height
    assert width_orig // 2 // 3 == point_cloud.width


def test_downsampled_chaining(point_cloud):
    # Remember original size
    height_orig = point_cloud.height
    width_orig = point_cloud.width

    # Perform downsampling
    point_cloud_new = point_cloud.downsampled("by2x2").downsampled("by3x3")

    # Check that a new object was returned and that the original is untouched
    assert isinstance(point_cloud_new, zivid.PointCloud)
    assert point_cloud_new is not point_cloud
    assert point_cloud.height == height_orig
    assert point_cloud.width == width_orig

    # Check result
    assert height_orig // 2 // 3 == point_cloud_new.height
    assert width_orig // 2 // 3 == point_cloud_new.width


def test_height_context_manager(frame):
    with frame.point_cloud() as point_cloud:
        _ = point_cloud.height
    with pytest.raises(RuntimeError):
        _ = point_cloud.height


def test_width_context_manager(frame):
    with frame.point_cloud() as point_cloud:
        _ = point_cloud.width
    with pytest.raises(RuntimeError):
        _ = point_cloud.width


def test_copy_data_context_manager(frame):
    with frame.point_cloud() as point_cloud:
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format=123)
    with pytest.raises(TypeError):
        point_cloud.copy_data()


def test_illegal_init(
    application,  # pylint: disable=unused-argument
):
    with pytest.raises(TypeError):
        zivid.PointCloud()  # pylint: disable=no-value-for-parameter

    with pytest.raises(TypeError):
        zivid.PointCloud("Should fail.")

    with pytest.raises(TypeError):
        zivid.PointCloud(123)


def test_copy_point_cloud(frame, transform):
    with frame.point_cloud() as point_cloud:
        with copy.copy(point_cloud) as point_cloud_copy:
            assert isinstance(point_cloud_copy, zivid.PointCloud)
            assert point_cloud_copy is not point_cloud
            assert_point_clouds_equal(point_cloud, point_cloud_copy)

            point_cloud.transform(transform)
            # shallow copy, transform should have affected both copies
            assert_point_clouds_equal(point_cloud, point_cloud_copy)


def test_deepcopy_point_cloud(frame, transform):
    with frame.point_cloud() as point_cloud:
        with copy.deepcopy(point_cloud) as point_cloud_copy:
            assert isinstance(point_cloud_copy, zivid.PointCloud)
            assert point_cloud_copy is not point_cloud
            assert_point_clouds_equal(point_cloud, point_cloud_copy)

            point_cloud.transform(transform)
            # deep copy, transform should not have affected the copy
            assert_point_clouds_not_equal(point_cloud, point_cloud_copy)


def test_clone_point_cloud(frame, transform):
    with frame.point_cloud() as point_cloud:
        with point_cloud.clone() as point_cloud_clone:
            assert isinstance(point_cloud_clone, zivid.PointCloud)
            assert point_cloud_clone is not point_cloud
            assert_point_clouds_equal(point_cloud, point_cloud_clone)

            point_cloud.transform(transform)
            # clone, transform should not have affected the clone
            assert_point_clouds_not_equal(point_cloud, point_cloud_clone)


def test_mask_with_numpy_boolean(point_cloud):
    """Test masking with numpy boolean array."""
    height, width = point_cloud.height, point_cloud.width

    # Create a boolean mask that masks out center region
    mask_array = np.ones((height, width), dtype=bool)
    mask_array[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = False

    # Get original point count
    original_xyz = point_cloud.copy_data("xyz")
    original_valid_points = np.sum(~np.isnan(original_xyz[:, :, 0]))

    # Apply mask in-place
    result = point_cloud.mask(mask_array)

    # Should return self for chaining
    assert result is point_cloud

    # Count valid points after masking
    masked_xyz = point_cloud.copy_data("xyz")
    masked_valid_points = np.sum(~np.isnan(masked_xyz[:, :, 0]))

    # Should have fewer valid points after masking
    assert masked_valid_points <= original_valid_points

    # Points where mask was True should now be NaN
    outside_center_slice_top = masked_xyz[: height // 4, :, 0]
    outside_center_slice_bottom = masked_xyz[3 * height // 4 :, :, 0]
    outside_center_slice_left = masked_xyz[:, : width // 4, 0]
    outside_center_slice_right = masked_xyz[:, 3 * width // 4 :, 0]
    assert np.all(np.isnan(outside_center_slice_top))
    assert np.all(np.isnan(outside_center_slice_bottom))
    assert np.all(np.isnan(outside_center_slice_left))
    assert np.all(np.isnan(outside_center_slice_right))


def test_mask_with_numpy_uint8(point_cloud):
    """Test masking with numpy uint8 array."""
    height, width = point_cloud.height, point_cloud.width

    # Create a uint8 mask (0 = keep, 1 = mask out)
    mask_array = np.zeros((height, width), dtype=np.uint8)
    mask_array[: height // 2, :] = 1  # Mask out top half

    # Get original data for comparison
    original_xyz = point_cloud.copy_data("xyz").copy()

    # Apply mask
    point_cloud.mask(mask_array)

    # Check results
    masked_xyz = point_cloud.copy_data("xyz")

    # Top half should be NaN
    top_half_masked = masked_xyz[: height // 2, :, 0]
    assert np.all(np.isnan(top_half_masked))

    # Bottom half should be unchanged where originally valid
    bottom_half_original = original_xyz[height // 2 :, :, 0]
    bottom_half_masked = masked_xyz[height // 2 :, :, 0]

    # Where original was valid, masked should be the same
    valid_original = ~np.isnan(bottom_half_original)
    np.testing.assert_array_equal(bottom_half_original[valid_original], bottom_half_masked[valid_original])


def test_mask_with_zivid_mask(point_cloud):
    """Test masking with zivid.Mask object."""
    height, width = point_cloud.height, point_cloud.width

    # Create mask using zivid.Mask (0 = keep, 1 = mask out)
    mask_array = np.ones((height, width), dtype=np.uint8)
    mask_array[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 0  # Only keep center region

    mask = zivid.Mask(mask_array)

    # Apply mask
    point_cloud.mask(mask)

    # Check results
    masked_xyz = point_cloud.copy_data("xyz")

    # Outside center region should be NaN
    # Top region
    top_region = masked_xyz[: height // 3, :, 0]
    assert np.all(np.isnan(top_region))

    # Bottom region
    bottom_region = masked_xyz[2 * height // 3 :, :, 0]
    assert np.all(np.isnan(bottom_region))

    # Left and right regions
    left_region = masked_xyz[:, : width // 3, 0]
    assert np.all(np.isnan(left_region))

    right_region = masked_xyz[:, 2 * width // 3 :, 0]
    assert np.all(np.isnan(right_region))


def test_masked_returns_new_instance(point_cloud):
    """Test that masked() returns a new PointCloud instance."""
    height, width = point_cloud.height, point_cloud.width

    # Create mask
    mask_array = np.ones((height, width), dtype=bool)
    mask_array[10:20, 10:20] = False

    # Get original data for comparison
    original_xyz = point_cloud.copy_data("xyz").copy()

    # Apply masked() - should return new instance
    masked_point_cloud = point_cloud.masked(mask_array)

    # Should be different instances
    assert masked_point_cloud is not point_cloud
    assert isinstance(masked_point_cloud, zivid.PointCloud)

    # Original should be unchanged
    current_xyz = point_cloud.copy_data("xyz")
    np.testing.assert_array_equal(original_xyz, current_xyz)

    # New instance should have mask applied
    masked_xyz = masked_point_cloud.copy_data("xyz")
    outside_masked_region_top = masked_xyz[:10, :, 0]
    outside_masked_region_bottom = masked_xyz[20:, :, 0]
    outside_masked_region_left = masked_xyz[:, :10, 0]
    outside_masked_region_right = masked_xyz[:, 20:, 0]
    assert np.all(np.isnan(outside_masked_region_top))
    assert np.all(np.isnan(outside_masked_region_bottom))
    assert np.all(np.isnan(outside_masked_region_left))
    assert np.all(np.isnan(outside_masked_region_right))

    # Clean up
    masked_point_cloud.release()


def test_mask_chaining(point_cloud):
    """Test that mask() returns self for method chaining."""
    height, width = point_cloud.height, point_cloud.width

    # Create simple masks
    mask1 = np.ones((height, width), dtype=bool)
    mask1[:10, :] = False  # Mask top 10 rows

    mask2 = np.ones((height, width), dtype=bool)
    mask2[-10:, :] = False  # Mask bottom 10 rows

    # Test chaining
    result = point_cloud.mask(mask1).mask(mask2)

    # Should return self
    assert result is point_cloud

    # Check that both masks were applied
    masked_xyz = point_cloud.copy_data("xyz")

    # Top and bottom should be NaN
    top_region = masked_xyz[:10, :, 0]
    bottom_region = masked_xyz[-10:, :, 0]

    assert np.all(np.isnan(top_region))
    assert np.all(np.isnan(bottom_region))


def test_mask_invalid_dimensions(point_cloud):
    """Test error handling for invalid mask dimensions."""
    height, width = point_cloud.height, point_cloud.width

    # Wrong dimensions
    wrong_mask = np.ones((height + 1, width), dtype=bool)

    with pytest.raises((ValueError, RuntimeError)):
        point_cloud.mask(wrong_mask)

    # Wrong number of dimensions
    wrong_dims = np.ones((height, width, 3), dtype=bool)

    with pytest.raises((ValueError, TypeError)):
        point_cloud.mask(wrong_dims)
