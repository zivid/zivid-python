import pytest


# These tests cannot be in the "test_application" module because that module depends on a
# zivid.Application fixture and these tests want to test the initialization and teardown of the
# zivid.Application class.


def test_init_with():
    import zivid

    with zivid.Application() as app:
        assert app
        assert isinstance(app, zivid.Application)


def test_release(file_camera_file):
    import zivid

    application = zivid.Application()
    assert application.create_file_camera(file_camera_file)
    application.release()
    with pytest.raises(RuntimeError):
        application.create_file_camera(file_camera_file)
