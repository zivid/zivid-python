import _zivid
import zivid


def _to_internal_suggest_settings_parameters_ambient_light_frequency(
    ambient_light_frequency,
):
    print("this is value:")
    print(ambient_light_frequency)
    print(type(ambient_light_frequency))
    try:
        if ambient_light_frequency is None:
            return (
                zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none
            )
        return zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._valid_values[
            ambient_light_frequency
        ]
    except Exception as ex:
        raise RuntimeError(
            "Unsupported value: {value}".format(value=ambient_light_frequency)
        ) from ex  # TODO


def to_internal_suggest_settings_parameters(suggest_settings_parameters):
    internal_suggest_settings_parameters = (
        _zivid.capture_assistant.SuggestSettingsParameters()
    )
    if suggest_settings_parameters.max_capture_time is not None:
        internal_suggest_settings_parameters.max_capture_time = _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
            suggest_settings_parameters.max_capture_time
        )
    else:
        internal_suggest_settings_parameters.max_capture_time = (
            _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime()
        )
    internal_suggest_settings_parameters.ambient_light_frequency = (
        suggest_settings_parameters.ambient_light_frequency
    )

    return internal_suggest_settings_parameters


import zivid


def to_suggest_settings_parameters(internal_suggest_settings_parameters):
    def _to_ambient_light_frequency(internal_ambient_light_frequency):

        return zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency()

    global to_ambient_light_frequency
    to_ambient_light_frequency = _to_ambient_light_frequency
    return zivid.capture_assistant.SuggestSettingsParameters(
        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._convert_to_non_internal_value(
            internal_suggest_settings_parameters.ambient_light_frequency
        ),
        max_capture_time=internal_suggest_settings_parameters.max_capture_time.value,
    )
