def _is_version(sut):
    return isinstance(sut, str) and len(sut) > 0


def test_module_version():
    import zivid

    assert _is_version(zivid.__version__)


def test_sdk_version():
    import zivid

    assert _is_version(zivid.sdk_version.SDKVersion.full)
    assert isinstance(zivid.sdk_version.SDKVersion.major, int)
    assert isinstance(zivid.sdk_version.SDKVersion.minor, int)
    assert isinstance(zivid.sdk_version.SDKVersion.patch, int)
    assert isinstance(zivid.sdk_version.SDKVersion.build, str)
