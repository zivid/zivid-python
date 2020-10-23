import pytest


def test_one_acquisition_in_list(file_camera):
    import zivid

    acquisitions = [zivid.Settings.Acquisition()]
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, list)
    with file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_five_acquisitions_in_list(file_camera):
    import zivid

    acquisitions = [
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
    ]
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, list)
    with file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_one_acquisition_in_tuple(file_camera):
    import zivid

    acquisitions = (zivid.Settings.Acquisition(),)
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, tuple)
    with file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_five_acquisition_in_tuple(file_camera):
    import zivid

    acquisitions = (
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
        zivid.Settings.Acquisition(),
    )
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, tuple)
    with file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_illegal_settings(file_camera):
    import zivid

    with pytest.raises(RuntimeError):
        file_camera.capture(zivid.Settings())

    with pytest.raises(TypeError):
        file_camera.capture([1, 2, 3, 4, 5])

    with pytest.raises(TypeError):
        file_camera.capture([zivid.Settings(), zivid.Settings(), 3])

    with pytest.raises(TypeError):
        file_camera.capture(file_camera.capture())


def test_empty_settings_list(file_camera):
    import _zivid

    with pytest.raises(TypeError):
        file_camera.capture([])
