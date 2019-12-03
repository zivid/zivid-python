def test_ambient_light_frequency():
    from zivid.captureassistant import AmbientLightFrequency

    assert str(AmbientLightFrequency.hz50) == "hz50"
    assert str(AmbientLightFrequency.hz60) == "hz60"
    assert str(AmbientLightFrequency.none) == "none"


def test_suggest_settings_parameters():
    import datetime
    from zivid.captureassistant import AmbientLightFrequency, SuggestSettingsParameters

    suggest_settings_parameters = SuggestSettingsParameters(
        budget=datetime.timedelta(milliseconds=1200),
        frequency=AmbientLightFrequency.hz50,
    )
    assert isinstance(suggest_settings_parameters.max_capture_time, datetime.timedelta)
    assert suggest_settings_parameters.max_capture_time == datetime.timedelta(
        seconds=1.2
    )
    assert isinstance(
        suggest_settings_parameters.ambient_light_frequency, AmbientLightFrequency
    )
    assert (
        suggest_settings_parameters.ambient_light_frequency
        == AmbientLightFrequency.hz50
    )


def test_suggest_settings_throws_if_budget_outside_range(file_camera):
    import datetime
    import zivid
    from zivid.captureassistant import AmbientLightFrequency, SuggestSettingsParameters

    # too small
    suggest_settings_parameters = SuggestSettingsParameters(
        budget=datetime.timedelta(milliseconds=100),
        frequency=AmbientLightFrequency.hz50,
    )
    with pytest.raises(RuntimeError):
        zivid.captureassistant.suggest_settings(
            file_camera, suggest_settings_parameters
        )

    # too big
    suggest_settings_parameters = SuggestSettingsParameters(
        datetime.timedelta(milliseconds=60000),
    )
    with pytest.raises(RuntimeError):
        zivid.captureassistant.suggest_settings(
            file_camera, suggest_settings_parameters
        )
