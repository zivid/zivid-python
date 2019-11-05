def test_point_cloud_to_array(point_cloud):
    import numpy as np

    np_array = point_cloud.to_array()
    assert np_array is not None
    assert isinstance(np_array, np.ndarray)


def test_to_rgb_image(point_cloud):
    import numpy as np

    np_array = point_cloud.to_array()
    image = np_array[["r", "g", "b"]]
    image = np.asarray([np_array["r"], np_array["g"], np_array["b"]])
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = image.astype(np.uint8)


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

    with frame.get_point_cloud() as point_cloud:
        point_cloud.height  # pylint: disable=pointless-statement
    with pytest.raises(RuntimeError):
        point_cloud.height  # pylint: disable=pointless-statement


def test_width_context_manager(frame):
    import pytest

    with frame.get_point_cloud() as point_cloud:
        point_cloud.width  # pylint: disable=pointless-statement
    with pytest.raises(RuntimeError):
        point_cloud.width  # pylint: disable=pointless-statement


def test_to_array_context_manager(frame):
    import pytest

    with frame.get_point_cloud() as point_cloud:
        point_cloud.to_array()
    with pytest.raises(RuntimeError):
        point_cloud.to_array()


def test_illegal_init(application):  # pylint: disable=unused-argument
    import pytest
    import zivid

    with pytest.raises(TypeError):
        zivid.PointCloud()  # pylint: disable=no-value-for-parameter

    with pytest.raises(ValueError):
        zivid.PointCloud("Should fail.")

    with pytest.raises(ValueError):
        zivid.PointCloud(123)
