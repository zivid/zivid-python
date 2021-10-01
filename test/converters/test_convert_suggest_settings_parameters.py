def test_to_internal_suggest_settings_parameters_to_suggest_settings_parameters_modified():
    from zivid.capture_assistant import SuggestSettingsParameters
    from zivid._capture_assistant_suggest_settings_parameters_converter import (
        to_capture_assistant_suggest_settings_parameters,
        to_internal_capture_assistant_suggest_settings_parameters,
    )

    modified_suggest_settings_parameters = SuggestSettingsParameters(
        ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.hz50
    )

    converted_suggest_settings_parameters = (
        to_capture_assistant_suggest_settings_parameters(
            to_internal_capture_assistant_suggest_settings_parameters(
                modified_suggest_settings_parameters
            )
        )
    )
    assert modified_suggest_settings_parameters == converted_suggest_settings_parameters
    assert isinstance(converted_suggest_settings_parameters, SuggestSettingsParameters)
    assert isinstance(modified_suggest_settings_parameters, SuggestSettingsParameters)


def test_to_internal_suggest_settings_parameters_to_suggest_settings_parameters_default():
    from zivid.capture_assistant import SuggestSettingsParameters
    from zivid._capture_assistant_suggest_settings_parameters_converter import (
        to_capture_assistant_suggest_settings_parameters,
        to_internal_capture_assistant_suggest_settings_parameters,
    )

    default_suggest_settings_parameters = SuggestSettingsParameters()
    converted_suggest_settings_parameters = (
        to_capture_assistant_suggest_settings_parameters(
            to_internal_capture_assistant_suggest_settings_parameters(
                default_suggest_settings_parameters
            )
        )
    )
    assert default_suggest_settings_parameters == converted_suggest_settings_parameters
    assert isinstance(converted_suggest_settings_parameters, SuggestSettingsParameters)
    assert isinstance(default_suggest_settings_parameters, SuggestSettingsParameters)


#
