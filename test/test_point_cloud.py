# pylint: disable=import-outside-toplevel


def test_point_cloud_copy_data(point_cloud):
    import pytest
    import numpy as np

    # Copy all possible formats
    xyz = point_cloud.copy_data("xyz")
    xyzw = point_cloud.copy_data("xyzw")
    xyzrgba = point_cloud.copy_data("xyzrgba")
    rgba = point_cloud.copy_data("rgba")
    depth = point_cloud.copy_data("z")
    snr = point_cloud.copy_data("snr")
    assert isinstance(xyz, np.ndarray)
    assert isinstance(xyzw, np.ndarray)
    assert isinstance(xyzrgba, np.ndarray)
    assert isinstance(rgba, np.ndarray)
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
    np.testing.assert_array_equal(xyz[:, :, 2], depth)
    np.testing.assert_array_equal(xyz[:, :, 0], xyzw[:, :, 0])
    np.testing.assert_array_equal(xyz[:, :, 1], xyzw[:, :, 1])
    np.testing.assert_array_equal(xyz[:, :, 2], xyzw[:, :, 2])


def test_point_cloud_rgba(point_cloud):
    import numpy as np

    xyzrgba = point_cloud.copy_data("xyzrgba")
    rgba = point_cloud.copy_data("rgba")

    assert rgba.shape == (point_cloud.height, point_cloud.width, 4)
    assert xyzrgba.shape == (point_cloud.height, point_cloud.width)
    assert rgba.dtype == np.uint8
    assert xyzrgba["r"].dtype == np.uint8
    assert xyzrgba["g"].dtype == np.uint8
    assert xyzrgba["b"].dtype == np.uint8
    assert xyzrgba["a"].dtype == np.uint8
    np.testing.assert_array_equal(rgba[:, :, 0], xyzrgba["r"])
    np.testing.assert_array_equal(rgba[:, :, 1], xyzrgba["g"])
    np.testing.assert_array_equal(rgba[:, :, 2], xyzrgba["b"])
    np.testing.assert_array_equal(rgba[:, :, 3], xyzrgba["a"])


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


def test_height_context_manager(frame):
    import pytest

    with frame.point_cloud() as point_cloud:
        point_cloud.height  # pylint: disable=pointless-statement
    with pytest.raises(RuntimeError):
        point_cloud.height  # pylint: disable=pointless-statement


def test_width_context_manager(frame):
    import pytest

    with frame.point_cloud() as point_cloud:
        point_cloud.width  # pylint: disable=pointless-statement
    with pytest.raises(RuntimeError):
        point_cloud.width  # pylint: disable=pointless-statement


def test_copy_data_context_manager(frame):
    import pytest

    with frame.point_cloud() as point_cloud:
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format="xyzrgba")
    with pytest.raises(RuntimeError):
        point_cloud.copy_data(data_format=123)
    with pytest.raises(TypeError):
        point_cloud.copy_data()


def test_illegal_init(application):  # pylint: disable=unused-argument
    import pytest
    import zivid

    with pytest.raises(TypeError):
        zivid.PointCloud()  # pylint: disable=no-value-for-parameter

    with pytest.raises(TypeError):
        zivid.PointCloud("Should fail.")

    with pytest.raises(TypeError):
        zivid.PointCloud(123)
