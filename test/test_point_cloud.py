def test_point_cloud_to_array(frame):
    import numpy as np

    point_cloud = frame.get_point_cloud()
    np_array = np.array(point_cloud)
    assert np_array is not None
    assert isinstance(np_array, np.ndarray)


def test_to_rgb_image(frame):
    import numpy as np

    point_cloud = frame.get_point_cloud()
    image = point_cloud[["r", "g", "b"]]
    image = np.asarray([point_cloud["r"], point_cloud["g"], point_cloud["b"]])
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = image.astype(np.uint8)
