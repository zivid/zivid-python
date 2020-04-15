# pylint: disable=import-outside-toplevel


def test_to_internal_settings_to_settings_modified():
    import datetime
    from zivid import Settings
    from zivid._settings_converter import to_settings, to_internal_settings

    modified_settings = Settings(
        acquisitions=[
            Settings.Acquisition(),
            Settings.Acquisition(exposure_time=datetime.timedelta(milliseconds=100)),
        ]
    )

    converted_settings = to_settings(to_internal_settings(modified_settings))
    assert modified_settings == converted_settings
    assert isinstance(converted_settings, Settings)
    assert isinstance(modified_settings, Settings)


def test_to_internal_settings_to_settings_default():
    from zivid import Settings
    from zivid._settings_converter import to_settings, to_internal_settings

    default_settings = Settings()
    converted_settings = to_settings(to_internal_settings(default_settings))
    assert default_settings == converted_settings
    assert isinstance(converted_settings, Settings)
    assert isinstance(default_settings, Settings)


#
