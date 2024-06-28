"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime
import _zivid


class SuggestSettingsParameters:

    class AmbientLightFrequency:

        hz50 = "hz50"
        hz60 = "hz60"
        none = "none"

        _valid_values = {
            "hz50": _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.hz50,
            "hz60": _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.hz60,
            "none": _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    def __init__(
        self,
        ambient_light_frequency=_zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency().value,
        max_capture_time=_zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime().value,
    ):

        if isinstance(
            ambient_light_frequency,
            _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.enum,
        ):
            self._ambient_light_frequency = _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
                ambient_light_frequency
            )
        elif isinstance(ambient_light_frequency, str):
            self._ambient_light_frequency = _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
                self.AmbientLightFrequency._valid_values[ambient_light_frequency]
            )
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(ambient_light_frequency)
                )
            )

        if isinstance(max_capture_time, (datetime.timedelta,)):
            self._max_capture_time = (
                _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
                    max_capture_time
                )
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                    value_type=type(max_capture_time)
                )
            )

    @property
    def ambient_light_frequency(self):
        if self._ambient_light_frequency.value is None:
            return None
        for key, internal_value in self.AmbientLightFrequency._valid_values.items():
            if internal_value == self._ambient_light_frequency.value:
                return key
        raise ValueError(
            "Unsupported value {value}".format(value=self._ambient_light_frequency)
        )

    @property
    def max_capture_time(self):
        return self._max_capture_time.value

    @ambient_light_frequency.setter
    def ambient_light_frequency(self, value):
        if isinstance(value, str):
            self._ambient_light_frequency = _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
                self.AmbientLightFrequency._valid_values[value]
            )
        elif isinstance(
            value,
            _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.enum,
        ):
            self._ambient_light_frequency = _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
                value
            )
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @max_capture_time.setter
    def max_capture_time(self, value):
        if isinstance(value, (datetime.timedelta,)):
            self._max_capture_time = (
                _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(value)
            )
        else:
            raise TypeError(
                "Unsupported type, expected: datetime.timedelta, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @classmethod
    def load(cls, file_name):
        return _to_capture_assistant_suggest_settings_parameters(
            _zivid.capture_assistant.SuggestSettingsParameters(str(file_name))
        )

    def save(self, file_name):
        _to_internal_capture_assistant_suggest_settings_parameters(self).save(
            str(file_name)
        )

    @classmethod
    def from_serialized(cls, value):
        return _to_capture_assistant_suggest_settings_parameters(
            _zivid.capture_assistant.SuggestSettingsParameters.from_serialized(
                str(value)
            )
        )

    def serialize(self):
        return _to_internal_capture_assistant_suggest_settings_parameters(
            self
        ).serialize()

    def __eq__(self, other):
        if (
            self._ambient_light_frequency == other._ambient_light_frequency
            and self._max_capture_time == other._max_capture_time
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_capture_assistant_suggest_settings_parameters(self))


def _to_capture_assistant_suggest_settings_parameters(
    internal_suggest_settings_parameters,
):
    return SuggestSettingsParameters(
        ambient_light_frequency=internal_suggest_settings_parameters.ambient_light_frequency.value,
        max_capture_time=internal_suggest_settings_parameters.max_capture_time.value,
    )


def _to_internal_capture_assistant_suggest_settings_parameters(
    suggest_settings_parameters,
):
    internal_suggest_settings_parameters = (
        _zivid.capture_assistant.SuggestSettingsParameters()
    )

    internal_suggest_settings_parameters.ambient_light_frequency = (
        _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency(
            suggest_settings_parameters._ambient_light_frequency.value
        )
    )
    internal_suggest_settings_parameters.max_capture_time = (
        _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
            suggest_settings_parameters.max_capture_time
        )
    )

    return internal_suggest_settings_parameters
