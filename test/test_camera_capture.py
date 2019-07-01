import pytest


def test_list_one_settings(file_camera):
    import zivid

    settings_collection = [zivid.Settings()]
    assert isinstance(settings_collection, list)
    with file_camera.capture(settings_collection) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.frame.Frame)


def test_list_five_settings(file_camera):
    import zivid

    settings_collection = [zivid.Settings() for _ in range(5)]
    assert isinstance(settings_collection, list)
    with file_camera.capture(settings_collection) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.frame.Frame)


def test_tuple_one_settings(file_camera):
    import zivid

    settings_collection = (zivid.Settings(),)
    assert isinstance(settings_collection, tuple)
    with file_camera.capture(settings_collection) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.frame.Frame)


def test_tuple_five_settings(file_camera):
    import zivid

    settings_collection = tuple([zivid.Settings() for _ in range(5)])
    assert isinstance(settings_collection, tuple)
    with file_camera.capture(settings_collection) as hdr_frame:
        assert hdr_frame
        assert isinstance(hdr_frame, zivid.frame.Frame)


def test_illegal_settings(file_camera):
    import zivid

    with pytest.raises(TypeError):
        file_camera.capture(zivid.Settings())

    with pytest.raises(AttributeError):
        file_camera.capture([1, 2, 3, 4, 5])

    with pytest.raises(AttributeError):
        file_camera.capture([zivid.Settings(), zivid.Settings(), 3])

    with pytest.raises(TypeError):
        file_camera.capture(file_camera.capture())


def test_empty_settings_list(file_camera):
    import _zivid

    with pytest.raises(RuntimeError):
        file_camera.capture([])
