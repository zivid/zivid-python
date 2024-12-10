import pytest


def test_point_cloud_copy_data(point_cloud):
    import numpy as np

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
    import numpy as np

    xyz = point_cloud.copy_data("xyz")
    xyzw = point_cloud.copy_data("xyzw")
    xyzrgba = point_cloud.copy_data("xyzrgba")
    xyzbgra = point_cloud.copy_data("xyzbgra")
    depth = point_cloud.copy_data("z")

    assert depth.shape == (point_cloud.height, point_cloud.width)
    assert xyz.shape == (point_cloud.height, point_cloud.width, 3)
    assert xyzw.shape == (point_cloud.height, point_cloud.width, 4)
    assert depth.dtype == np.float32
    assert xyz.dtype == np.float32
    assert xyzw.dtype == np.float32
    assert xyzrgba["x"].dtype == np.float32
    assert xyzrgba["y"].dtype == np.float32
    assert xyzrgba["z"].dtype == np.float32
    assert xyzbgra["x"].dtype == np.float32
    assert xyzbgra["y"].dtype == np.float32
    assert xyzbgra["z"].dtype == np.float32
    np.testing.assert_array_equal(xyz[:, :, 2], depth)
    np.testing.assert_array_equal(xyz[:, :, 0], xyzw[:, :, 0])
    np.testing.assert_array_equal(xyz[:, :, 1], xyzw[:, :, 1])
    np.testing.assert_array_equal(xyz[:, :, 2], xyzw[:, :, 2])


def test_point_cloud_rgba(point_cloud):
    import numpy as np

    xyzrgba = point_cloud.copy_data("xyzrgba")
    xyzbgra = point_cloud.copy_data("xyzbgra")
    rgba = point_cloud.copy_data("rgba")
    bgra = point_cloud.copy_data("bgra")
    srgb = point_cloud.copy_data("srgb")

    assert rgba.shape == (point_cloud.height, point_cloud.width, 4)
    assert bgra.shape == (point_cloud.height, point_cloud.width, 4)
    assert srgb.shape == (point_cloud.height, point_cloud.width, 4)
    assert xyzrgba.shape == (point_cloud.height, point_cloud.width)
    assert xyzbgra.shape == (point_cloud.height, point_cloud.width)
    assert rgba.dtype == np.uint8
    assert bgra.dtype == np.uint8
    assert srgb.dtype == np.uint8
    assert xyzrgba["r"].dtype == np.uint8
    assert xyzrgba["g"].dtype == np.uint8
    assert xyzrgba["b"].dtype == np.uint8
    assert xyzrgba["a"].dtype == np.uint8
    np.testing.assert_array_equal(rgba[:, :, 0], xyzrgba["r"])
    np.testing.assert_array_equal(rgba[:, :, 1], xyzrgba["g"])
    np.testing.assert_array_equal(rgba[:, :, 2], xyzrgba["b"])
    np.testing.assert_array_equal(rgba[:, :, 3], xyzrgba["a"])
    assert xyzbgra["b"].dtype == np.uint8
    assert xyzbgra["g"].dtype == np.uint8
    assert xyzbgra["r"].dtype == np.uint8
    assert xyzbgra["a"].dtype == np.uint8
    np.testing.assert_array_equal(bgra[:, :, 0], xyzbgra["b"])
    np.testing.assert_array_equal(bgra[:, :, 1], xyzbgra["g"])
    np.testing.assert_array_equal(bgra[:, :, 2], xyzbgra["r"])
    np.testing.assert_array_equal(bgra[:, :, 3], xyzbgra["a"])
    np.testing.assert_array_equal(xyzbgra["r"], xyzrgba["r"])
    np.testing.assert_array_equal(xyzbgra["g"], xyzrgba["g"])
    np.testing.assert_array_equal(xyzbgra["b"], xyzrgba["b"])
    np.testing.assert_array_equal(xyzbgra["a"], xyzrgba["a"])
    np.testing.assert_array_equal(bgra[:, :, 0], rgba[:, :, 2])
    np.testing.assert_array_equal(bgra[:, :, 1], rgba[:, :, 1])
    np.testing.assert_array_equal(bgra[:, :, 2], rgba[:, :, 0])
    np.testing.assert_array_equal(bgra[:, :, 3], rgba[:, :, 3])


def test_point_cloud_copy_image(point_cloud):
    import numpy as np
    import zivid

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
    import numpy as np

    normals = point_cloud.copy_data("normals")

    assert normals.shape == (point_cloud.height, point_cloud.width, 3)
    assert normals.dtype == np.float32

    normals_flat = normals.reshape(normals.shape[0] * normals.shape[1], 3)
    non_nan_normals = normals_flat[~np.isnan(normals_flat[:, 0])]
    vector_lengths = np.linalg.norm(non_nan_normals, axis=1)
    np.testing.assert_array_almost_equal(vector_lengths, 1.0)


def test_point_cloud_snr(point_cloud):
    import numpy as np

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
    import numpy as np

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
    import zivid

    # Get points before and after transform
    xyzw_before = point_cloud.copy_data("xyzw")
    point_cloud_returned = point_cloud.transform(transform)
    xyzw_after = point_cloud.copy_data("xyzw")

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the transformation was actually applied
    _validate_transformation(xyzw_before, xyzw_after, transform)


def test_transform_chaining(point_cloud, transform):
    import zivid
    import numpy as np

    # Get points before and after transform
    xyzw_before = point_cloud.copy_data("xyzw")
    point_cloud_returned = point_cloud.transform(transform).transform(transform)
    xyzw_after = point_cloud.copy_data("xyzw")

    # Check that return value is just a reference to the original object
    assert isinstance(point_cloud_returned, zivid.PointCloud)
    assert point_cloud_returned is point_cloud

    # Check that the transformation was actually applied
    _validate_transformation(xyzw_before, xyzw_after, np.dot(transform, transform))


def test_downsampling_enum():
    import zivid

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
    import zivid

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
    import zivid

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
    import zivid

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
    import zivid

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
        point_cloud.height
    with pytest.raises(RuntimeError):
        point_cloud.height


def test_width_context_manager(frame):
    with frame.point_cloud() as point_cloud:
        point_cloud.width
    with pytest.raises(RuntimeError):
        point_cloud.width


def test_copy_data_context_manager(frame):
    with frame.point_cloud() as point_cloud:
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format=123)
    with pytest.raises(TypeError):
        point_cloud.copy_data()


def test_illegal_init(application):
    import zivid

    with pytest.raises(TypeError):
        zivid.PointCloud()  # pylint: disable=no-value-for-parameter

    with pytest.raises(TypeError):
        zivid.PointCloud("Should fail.")

    with pytest.raises(TypeError):
        zivid.PointCloud(123)
