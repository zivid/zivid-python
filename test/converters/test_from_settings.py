def test_to_internal_settings_to_settings_modified(random_settings):
    from zivid import Settings
    from zivid._settings_converter import to_settings, to_internal_settings

    converted_settings = to_settings(to_internal_settings(random_settings))
    assert random_settings == converted_settings
    assert isinstance(converted_settings, Settings)
    assert isinstance(random_settings, Settings)


def test_to_internal_settings_to_settings_default():
    from zivid import Settings
    from zivid._settings_converter import to_settings, to_internal_settings

    default_settings = Settings()
    converted_settings = to_settings(to_internal_settings(default_settings))

    assert default_settings == converted_settings
    assert isinstance(converted_settings, Settings)
    assert isinstance(default_settings, Settings)
