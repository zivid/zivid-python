import pytest
import zivid

# These tests cannot be in the "test_application" module because that module depends on a
# zivid.Application fixture and these tests want to test the initialization and teardown of the
# zivid.Application class.


def test_init_with():
    with zivid.Application() as app:
        assert app
        assert isinstance(app, zivid.Application)


def test_release(file_camera_file):
    application = zivid.Application()
    assert application.create_file_camera(file_camera_file)
    application.release()
    with pytest.raises(RuntimeError):
        application.create_file_camera(file_camera_file)


def test_delete(file_camera_file):
    app1 = zivid.Application()
    assert app1.create_file_camera(file_camera_file)
    del app1

    app2 = zivid.Application()
    assert app2.create_file_camera(file_camera_file)
