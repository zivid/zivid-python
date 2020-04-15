"""Auto generated, do not edit"""
import _zivid
import zivid


def to_capture_assistant_suggest_settings_parameters_ambient_light_frequency(
    internal_ambient_light_frequency,
):
    for (
        key,
        value,
    ) in (
        zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._valid_values.items()
    ):
        if value == internal_ambient_light_frequency.value:
            return key
    raise ValueError(
        "Unsupported value: {value}".format(value=internal_ambient_light_frequency)
    )


def to_capture_assistant_suggest_settings_parameters(
    internal_suggest_settings_parameters,
):
    return zivid.capture_assistant.SuggestSettingsParameters(
        ambient_light_frequency=to_capture_assistant_suggest_settings_parameters_ambient_light_frequency(
            internal_suggest_settings_parameters.ambient_light_frequency
        ),
        max_capture_time=internal_suggest_settings_parameters.max_capture_time.value,
    )


def to_internal_capture_assistant_suggest_settings_parameters(
    suggest_settings_parameters,
):
    internal_suggest_settings_parameters = (
        _zivid.capture_assistant.SuggestSettingsParameters()
    )

    internal_suggest_settings_parameters.ambient_light_frequency = to_internal_capture_assistant_suggest_settings_parameters_ambient_light_frequency(
        suggest_settings_parameters.ambient_light_frequency
    )
    internal_suggest_settings_parameters.max_capture_time = _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
        suggest_settings_parameters.max_capture_time
    )

    return internal_suggest_settings_parameters


def to_internal_capture_assistant_suggest_settings_parameters_ambient_light_frequency(
    ambient_light_frequency,
):
    try:
        return _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
            zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._valid_values[
                ambient_light_frequency
            ]
        )
    except Exception as ex:
        raise ValueError(
            "Unsupported value: {value}".format(value=ambient_light_frequency)
        ) from ex  # TODO


# """Auto generated, do not edit"""
# import zivid
#
#
# def to_suggest_settings_parameters(internal_suggest_settings_parameters):
#     def _to_ambient_light_frequency(internal_ambient_light_frequency):
#         for (
#             key,
#             value,
#         ) in (
#             zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._valid_values.items()
#         ):
#             if value == internal_ambient_light_frequency.value:
#                 return key
#         raise ValueError(
#             "Unsupported value: {value}".format(value=internal_ambient_light_frequency)
#         )
#
#         # return zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency()
#
#     global to_ambient_light_frequency
#     to_ambient_light_frequency = _to_ambient_light_frequency
#     return zivid.capture_assistant.SuggestSettingsParameters(
#         ambient_light_frequency=_to_ambient_light_frequency(
#             internal_suggest_settings_parameters.ambient_light_frequency
#         ),
#         max_capture_time=internal_suggest_settings_parameters.max_capture_time.value,
#     )


# def to_suggest_settings_parameters(internal_suggest_settings_parameters):
#     def _to_ambient_light_frequency(internal_ambient_light_frequency):
#
#         return zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency()
#
#     global to_ambient_light_frequency
#     to_ambient_light_frequency = _to_ambient_light_frequency
#     return zivid.capture_assistant.SuggestSettingsParameters(
#         ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency._convert_to_non_internal_value(
#             internal_suggest_settings_parameters.ambient_light_frequency
#         ),
#         max_capture_time=internal_suggest_settings_parameters.max_capture_time.value,
#     )


# def to_internal_suggest_settings_parameters(suggest_settings_parameters):
#     internal_suggest_settings_parameters = (
#         _zivid.capture_assistant.SuggestSettingsParameters()
#     )
#
#     global to_internal_ambient_light_frequency
#     to_internal_ambient_light_frequency = (
#         _to_internal_suggest_settings_parameters_ambient_light_frequency
#     )
#
#     internal_suggest_settings_parameters.max_capture_time = _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
#         suggest_settings_parameters.max_capture_time
#     )
#
#     internal_suggest_settings_parameters.ambient_light_frequency = _to_internal_suggest_settings_parameters_ambient_light_frequency(
#         suggest_settings_parameters.ambient_light_frequency
#     )
#     return internal_suggest_settings_parameters


# def to_internal_suggest_settings_parameters(suggest_settings_parameters):
#     internal_suggest_settings_parameters = _zivid.SuggestSettingsParameters()
#
#     def _to_internal_ambient_light_frequency(ambient_light_frequency):
#         internal_ambient_light_frequency = (
#             _zivid.SuggestSettingsParameters.AmbientLightFrequency()
#         )
#
#         return internal_ambient_light_frequency
#
#     global to_internal_ambient_light_frequency
#     to_internal_ambient_light_frequency = (
#         _to_internal_suggest_settings_parameters_ambient_light_frequency
#     )
#
#     internal_suggest_settings_parameters.max_capture_time = _zivid.SuggestSettingsParameters.MaxCaptureTime(
#         suggest_settings_parameters.max_capture_time
#     )
#
#     internal_suggest_settings_parameters.ambient_light_frequency = _to_internal_suggest_settings_parameters_ambient_light_frequency(
#         suggest_settings_parameters.ambient_light_frequency
#     )
#     return internal_suggest_settings_parameters
