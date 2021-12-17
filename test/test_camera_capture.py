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


def test_diagnostics_capture(file_camera):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os.path
    import zivid

    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
    settings.diagnostics.enabled = False
    frame_diagnostics_off = file_camera.capture(settings)
    settings.diagnostics.enabled = True
    frame_diagnostics_on = file_camera.capture(settings)

    with TemporaryDirectory() as tmpdir:
        file_diagnostics_off = Path(tmpdir) / "diagnostics_off.zdf"
        file_diagnostics_on = Path(tmpdir) / "diagnostics_on.zdf"
        frame_diagnostics_off.save(file_diagnostics_off)
        frame_diagnostics_on.save(file_diagnostics_on)

        # Diagnostics ON should lead to a significantly larger file size.
        mb_diagnostics_off = os.path.getsize(str(file_diagnostics_off)) / 1e6
        mb_diagnostics_on = os.path.getsize(str(file_diagnostics_on)) / 1e6
        assert mb_diagnostics_on > (mb_diagnostics_off + 1.0)
