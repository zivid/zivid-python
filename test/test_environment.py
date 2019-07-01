def test_data_path(application):  # pylint: disable=unused-argument
    import zivid
    import pathlib
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as temp_dir:
        os.environ["ZIVID_DATA"] = temp_dir
        data_path = zivid.environment.data_path()
        assert isinstance(data_path, pathlib.Path)
        assert data_path == pathlib.Path(temp_dir)
