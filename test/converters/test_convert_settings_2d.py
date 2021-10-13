def test_to_internal_settings_to_settings_modified():
    import datetime
    from zivid import Settings2D
    from zivid.settings_2d import _to_settings2d, _to_internal_settings2d

    modified_settings = Settings2D(
        acquisitions=[
            Settings2D.Acquisition(),
            Settings2D.Acquisition(exposure_time=datetime.timedelta(milliseconds=100)),
        ]
    )
    converted_settings = _to_settings2d(_to_internal_settings2d(modified_settings))
    assert modified_settings == converted_settings
    assert isinstance(converted_settings, Settings2D)
    assert isinstance(modified_settings, Settings2D)


def test_to_internal_settings_to_settings_default():
    from zivid import Settings2D
    from zivid.settings_2d import _to_settings2d, _to_internal_settings2d

    default_settings = Settings2D()
    converted_settings = _to_settings2d(_to_internal_settings2d(default_settings))
    assert default_settings == converted_settings
    assert isinstance(converted_settings, Settings2D)
    assert isinstance(default_settings, Settings2D)


#
