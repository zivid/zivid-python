# pylint: disable=import-outside-toplevel


def test_to_internal_settings_to_settings_modified():
    import datetime
    from zivid import Settings2D
    from zivid._settings_2d_converter import to_settings_2d, to_internal_settings_2d

    modified_settings = Settings2D(
        acquisitions=[
            Settings2D.Acquisition(),
            Settings2D.Acquisition(exposure_time=datetime.timedelta(milliseconds=100)),
        ]
    )

    converted_settings = to_settings_2d(to_internal_settings_2d(modified_settings))
    assert modified_settings == converted_settings
    assert isinstance(converted_settings, Settings2D)
    assert isinstance(modified_settings, Settings2D)


def test_to_internal_settings_to_settings_default():
    from zivid import Settings2D
    from zivid._settings_2d_converter import to_settings_2d, to_internal_settings_2d

    default_settings = Settings2D()
    converted_settings = to_settings_2d(to_internal_settings_2d(default_settings))
    assert default_settings == converted_settings
    assert isinstance(converted_settings, Settings2D)
    assert isinstance(default_settings, Settings2D)


#
