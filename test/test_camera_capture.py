import pytest


def test_capture_2d_3d_one_2d_and_one_3d(shared_file_camera):
    import zivid

    acquisitions3d = [zivid.Settings.Acquisition()]
    acquisitions2d = [zivid.Settings2D.Acquisition()]
    settings = zivid.Settings(
        acquisitions=acquisitions3d, color=zivid.Settings2D(acquisitions=acquisitions2d)
    )

    with shared_file_camera.capture_2d_3d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)
        assert len(frame.settings.acquisitions) == 1
        assert frame.settings.color
        assert len(frame.settings.color.acquisitions) == 1
        assert frame.frame_2d()
        assert isinstance(frame.frame_2d(), zivid.Frame2D)


def test_capture_2d_3d_two_2d_and_one_3d(shared_file_camera):
    import zivid

    acquisitions3d = [zivid.Settings.Acquisition()]
    acquisitions2d = [zivid.Settings2D.Acquisition(), zivid.Settings2D.Acquisition()]
    settings = zivid.Settings(
        acquisitions=acquisitions3d, color=zivid.Settings2D(acquisitions=acquisitions2d)
    )

    with shared_file_camera.capture_2d_3d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)
        assert len(frame.settings.acquisitions) == 1
        assert frame.settings.color
        assert len(frame.settings.color.acquisitions) == 2
        assert frame.frame_2d()
        assert isinstance(frame.frame_2d(), zivid.Frame2D)


def test_capture_2d_3d_one_2d_and_two_3d(shared_file_camera):
    import zivid

    acquisitions3d = [zivid.Settings.Acquisition(), zivid.Settings.Acquisition()]
    acquisitions2d = [zivid.Settings2D.Acquisition()]
    settings = zivid.Settings(
        acquisitions=acquisitions3d, color=zivid.Settings2D(acquisitions=acquisitions2d)
    )

    with shared_file_camera.capture_2d_3d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)
        assert len(frame.settings.acquisitions) == 2
        assert frame.settings.color
        assert len(frame.settings.color.acquisitions) == 1
        assert frame.frame_2d()
        assert isinstance(frame.frame_2d(), zivid.Frame2D)


def test_capture_2d_3d_two_2d_and_two_3d(shared_file_camera):
    import zivid

    acquisitions3d = [zivid.Settings.Acquisition(), zivid.Settings.Acquisition()]
    acquisitions2d = [zivid.Settings2D.Acquisition(), zivid.Settings2D.Acquisition()]
    settings = zivid.Settings(
        acquisitions=acquisitions3d, color=zivid.Settings2D(acquisitions=acquisitions2d)
    )

    with shared_file_camera.capture_2d_3d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)
        assert len(frame.settings.acquisitions) == 2
        assert frame.settings.color
        assert len(frame.settings.color.acquisitions) == 2
        assert frame.frame_2d()
        assert isinstance(frame.frame_2d(), zivid.Frame2D)


def test_capture_3d_one_acquisition(shared_file_camera):
    import zivid

    acquisitions = [zivid.Settings.Acquisition()]
    settings = zivid.Settings(acquisitions=acquisitions)
    with shared_file_camera.capture_3d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)
        assert len(frame.settings.acquisitions) == 1
        assert frame.settings.color is None
        assert frame.frame_2d() is None


def test_capture_2d_with_settings_2d(shared_file_camera):
    import zivid

    acquisitions = [zivid.Settings2D.Acquisition()]
    settings = zivid.Settings2D(acquisitions=acquisitions)
    with shared_file_camera.capture_2d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.Frame2D)
        assert len(frame.settings.acquisitions) == 1


def test_capture_2d_with_settings(shared_file_camera):
    import zivid

    acquisitions = [zivid.Settings2D.Acquisition()]
    settings = zivid.Settings(color=zivid.Settings2D(acquisitions=acquisitions))
    with shared_file_camera.capture_2d(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.Frame2D)
        assert len(frame.settings.acquisitions) == 1


def test_one_acquisition_in_list(shared_file_camera):
    import zivid

    acquisitions = [zivid.Settings.Acquisition()]
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, list)
    with shared_file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_five_acquisitions_in_list(shared_file_camera):
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
    with shared_file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_one_acquisition_in_tuple(shared_file_camera):
    import zivid

    acquisitions = (zivid.Settings.Acquisition(),)
    settings = zivid.Settings(acquisitions=acquisitions)
    assert isinstance(acquisitions, tuple)
    with shared_file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_five_acquisition_in_tuple(shared_file_camera):
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
    with shared_file_camera.capture(settings) as frame:
        assert frame
        assert isinstance(frame, zivid.frame.Frame)


def test_illegal_settings(shared_file_camera):
    import zivid

    with pytest.raises(RuntimeError):
        shared_file_camera.capture(zivid.Settings())

    with pytest.raises(TypeError):
        shared_file_camera.capture([1, 2, 3, 4, 5])

    with pytest.raises(TypeError):
        shared_file_camera.capture([zivid.Settings(), zivid.Settings(), 3])

    with pytest.raises(TypeError):
        shared_file_camera.capture(shared_file_camera.capture())


def test_empty_settings_list(shared_file_camera):
    import _zivid

    with pytest.raises(TypeError):
        shared_file_camera.capture([])


def test_diagnostics_capture(shared_file_camera):
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import os.path
    import zivid

    settings = zivid.Settings()
    settings.acquisitions.append(zivid.Settings.Acquisition())
    settings.diagnostics.enabled = False
    frame_diagnostics_off = shared_file_camera.capture(settings)
    settings.diagnostics.enabled = True
    frame_diagnostics_on = shared_file_camera.capture(settings)

    with TemporaryDirectory() as tmpdir:
        file_diagnostics_off = Path(tmpdir) / "diagnostics_off.zdf"
        file_diagnostics_on = Path(tmpdir) / "diagnostics_on.zdf"
        frame_diagnostics_off.save(file_diagnostics_off)
        frame_diagnostics_on.save(file_diagnostics_on)

        # Diagnostics ON should lead to a significantly larger file size.
        mb_diagnostics_off = os.path.getsize(str(file_diagnostics_off)) / 1e6
        mb_diagnostics_on = os.path.getsize(str(file_diagnostics_on)) / 1e6
        assert mb_diagnostics_on > (mb_diagnostics_off + 1.0)
