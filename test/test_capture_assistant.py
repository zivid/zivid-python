import datetime
import pytest


def test_ambient_light_frequency():
    from zivid.capture_assistant import AmbientLightFrequency

    assert str(AmbientLightFrequency.hz50) == "hz50"
    assert str(AmbientLightFrequency.hz60) == "hz60"
    assert str(AmbientLightFrequency.none) == "none"


def test_suggest_settings_parameters():
    from zivid.capture_assistant import AmbientLightFrequency, SuggestSettingsParameters

    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=AmbientLightFrequency.hz50,
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
    import zivid
    from zivid.capture_assistant import AmbientLightFrequency, SuggestSettingsParameters

    # too small
    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=100),
        ambient_light_frequency=AmbientLightFrequency.hz50,
    )
    with pytest.raises(RuntimeError):
        zivid.capture_assistant.suggest_settings(
            file_camera, suggest_settings_parameters
        )

    # too big
    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=60000)
    )
    with pytest.raises(RuntimeError):
        zivid.capture_assistant.suggest_settings(
            file_camera, suggest_settings_parameters
        )
