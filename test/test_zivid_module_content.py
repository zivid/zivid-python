def test_import_zivid_globals_changes():
    before = sorted(globals().keys())
    import zivid  # pylint: disable=unused-import, possibly-unused-variable, import-outside-toplevel  # noqa: F401

    after = sorted(globals().keys())
    assert before == after


def test_import_zivid_globals_locals():
    expected_changes = sorted(["before", "zivid"])
    before = sorted(locals().keys())
    import zivid  # pylint: disable=unused-import, possibly-unused-variable, import-outside-toplevel  # noqa: F401

    after = sorted(locals().keys())
    assert sorted(before + expected_changes) == after


def test_version():
    import _zivid  # pylint: disable=import-outside-toplevel
    import zivid  # pylint: disable=import-outside-toplevel

    assert zivid.__version__
    assert _zivid.__version__
    assert zivid.__version__ == _zivid.__version__
