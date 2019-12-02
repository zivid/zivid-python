def test_ambient_light_frequency():
    from zivid.captureassistant import AmbientLightFrequency

    assert str(AmbientLightFrequency.hz50) == 'hz50'
    assert str(AmbientLightFrequency.hz60) == 'hz60'
    assert str(AmbientLightFrequency.none) == 'none'


def test_suggest_settings_parameters():
    import datetime
    from zivid.captureassistant import AmbientLightFrequency, SuggestSettingsParameters

    suggest_settings_parameters = \
        SuggestSettingsParameters(
            budget=datetime.timedelta(milliseconds=1200),
            frequency=AmbientLightFrequency.hz50)
    assert isinstance(suggest_settings_parameters.max_capture_time, datetime.timedelta)
    assert suggest_settings_parameters.max_capture_time == datetime.timedelta(seconds=1.2)
    assert isinstance(suggest_settings_parameters.ambient_light_frequency, AmbientLightFrequency)
    assert suggest_settings_parameters.ambient_light_frequency == AmbientLightFrequency.hz50


def test_suggest_settings(file_camera):
    import datetime
    import zivid
    from zivid.captureassistant import AmbientLightFrequency, SuggestSettingsParameters

    suggest_settings_parameters = \
        SuggestSettingsParameters(
            budget=datetime.timedelta(milliseconds=1200),
            frequency=AmbientLightFrequency.hz50)
    suggested_settings = zivid.captureassistant.suggest_settings(file_camera, suggest_settings_parameters)
    assert suggested_settings
    assert isinstance(suggested_settings, list)
    isinstance(suggested_settings[0], zivid.settings.Settings)

    hdr_frame = zivid.hdr.capture(file_camera, suggested_settings)
    assert hdr_frame
    assert isinstance(hdr_frame, zivid.frame.Frame)
