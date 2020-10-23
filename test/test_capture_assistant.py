import pytest


def test_ambient_light_frequency():
    from zivid.capture_assistant import SuggestSettingsParameters

    assert str(SuggestSettingsParameters.AmbientLightFrequency.hz50) == "hz50"
    assert str(SuggestSettingsParameters.AmbientLightFrequency.hz60) == "hz60"
    assert str(SuggestSettingsParameters.AmbientLightFrequency.none) == "none"

    assert SuggestSettingsParameters.AmbientLightFrequency.hz50 == "hz50"
    assert SuggestSettingsParameters.AmbientLightFrequency.hz60 == "hz60"
    assert SuggestSettingsParameters.AmbientLightFrequency.none == "none"


def test_suggest_settings_parameters():
    import datetime
    from zivid.capture_assistant import SuggestSettingsParameters

    # Use constructor
    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.hz50,
    )
    assert isinstance(suggest_settings_parameters.max_capture_time, datetime.timedelta)
    assert suggest_settings_parameters.max_capture_time == datetime.timedelta(
        seconds=1.2
    )
    assert isinstance(suggest_settings_parameters.ambient_light_frequency, str)
    assert (
        suggest_settings_parameters.ambient_light_frequency
        == SuggestSettingsParameters.AmbientLightFrequency.hz50
    )

    # Use setters
    new_time = datetime.timedelta(milliseconds=1800)
    new_freq = "hz60"
    suggest_settings_parameters.max_capture_time = new_time
    assert isinstance(suggest_settings_parameters.max_capture_time, datetime.timedelta)
    assert suggest_settings_parameters.max_capture_time == new_time
    suggest_settings_parameters.ambient_light_frequency = new_freq
    assert isinstance(suggest_settings_parameters.ambient_light_frequency, str)
    assert suggest_settings_parameters.ambient_light_frequency == new_freq


def test_suggest_settings_throws_if_budget_outside_range():
    import datetime
    from zivid.capture_assistant import SuggestSettingsParameters

    # too small
    with pytest.raises(IndexError):
        SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=100),
            ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.hz50,
        )
    # too big
    with pytest.raises(IndexError):
        SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=60000)
        )


def test_init_max_capture_time():
    import datetime
    from zivid.capture_assistant import SuggestSettingsParameters

    suggested_settings = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1000)
    )
    max_capture_time = suggested_settings.max_capture_time
    assert max_capture_time is not None
    assert isinstance(max_capture_time, datetime.timedelta)
    assert max_capture_time == datetime.timedelta(milliseconds=1000)


def test_default_ambient_light_frequency():
    import datetime
    from zivid.capture_assistant import SuggestSettingsParameters

    suggested_settings = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=250)
    )
    ambient_light_frequency = suggested_settings.ambient_light_frequency
    assert ambient_light_frequency is not None
    assert isinstance(ambient_light_frequency, str)
    assert (
        ambient_light_frequency == SuggestSettingsParameters.AmbientLightFrequency.none
    )


def test_set_ambient_light_frequency():
    from zivid.capture_assistant import SuggestSettingsParameters

    suggested_settings = SuggestSettingsParameters()

    suggested_settings.ambient_light_frequency = (
        SuggestSettingsParameters.AmbientLightFrequency.hz50
    )
    assert (
        suggested_settings.ambient_light_frequency
        == SuggestSettingsParameters.AmbientLightFrequency.hz50
    )
    assert suggested_settings.ambient_light_frequency == "hz50"
    assert isinstance(suggested_settings.ambient_light_frequency, str)

    suggested_settings.ambient_light_frequency = (
        SuggestSettingsParameters.AmbientLightFrequency.hz60
    )
    assert (
        suggested_settings.ambient_light_frequency
        == SuggestSettingsParameters.AmbientLightFrequency.hz60
    )
    assert suggested_settings.ambient_light_frequency == "hz60"
    assert isinstance(suggested_settings.ambient_light_frequency, str)

    suggested_settings.ambient_light_frequency = (
        SuggestSettingsParameters.AmbientLightFrequency.none
    )
    assert (
        suggested_settings.ambient_light_frequency
        == SuggestSettingsParameters.AmbientLightFrequency.none
    )
    assert suggested_settings.ambient_light_frequency == "none"
    assert isinstance(suggested_settings.ambient_light_frequency, str)


def test_suggest_settings_str():
    from zivid.capture_assistant import SuggestSettingsParameters

    string = str(SuggestSettingsParameters())
    assert string is not None
    assert isinstance(string, str)


def test_ambient_light_frequency_str():
    from zivid.capture_assistant import SuggestSettingsParameters

    string = str(SuggestSettingsParameters().ambient_light_frequency)
    assert string is not None
    assert isinstance(string, str)
