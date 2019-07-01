def test_import_zivid_globals_changes():
    before = sorted(globals().keys())
    import zivid  # pylint: disable=unused-import

    after = sorted(globals().keys())
    assert before == after


def test_import_zivid_globals_locals():
    expected_changes = sorted(["before", "zivid"])
    before = sorted(locals().keys())
    import zivid  # pylint: disable=possibly-unused-variable

    after = sorted(locals().keys())
    assert sorted(before + expected_changes) == after


def test_version():
    import zivid
    import _zivid

    assert zivid.__version__
    assert _zivid.__version__
    assert zivid.__version__ == _zivid.__version__
