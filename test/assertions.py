import numpy as np


def _assert_xyzrgba_equal(a, b):
    # Also works for xyzbgra

    for key in "xyzrgba":
        np.testing.assert_array_equal(a[key], b[key])


def assert_point_clouds_equal(a, b):
    assert b.height == a.height
    assert b.width == a.width

    np.testing.assert_array_equal(a.copy_data("xyz"), b.copy_data("xyz"))
    np.testing.assert_array_equal(a.copy_data("xyzw"), b.copy_data("xyzw"))
    _assert_xyzrgba_equal(
        a.copy_data("xyzrgba"),
        b.copy_data("xyzrgba"),
    )
    _assert_xyzrgba_equal(
        a.copy_data("xyzbgra"),
        b.copy_data("xyzbgra"),
    )
    np.testing.assert_array_equal(a.copy_data("rgba"), b.copy_data("rgba"))
    np.testing.assert_array_equal(a.copy_data("bgra"), b.copy_data("bgra"))
    np.testing.assert_array_equal(a.copy_data("srgb"), b.copy_data("srgb"))
    np.testing.assert_array_equal(a.copy_data("normals"), b.copy_data("normals"))
    np.testing.assert_array_equal(a.copy_data("z"), b.copy_data("z"))
    np.testing.assert_array_equal(a.copy_data("snr"), b.copy_data("snr"))


def assert_point_clouds_not_equal(a, b):
    try:
        assert_point_clouds_equal(a, b)
    except AssertionError:
        pass
    else:
        raise AssertionError("Point clouds are equal")
