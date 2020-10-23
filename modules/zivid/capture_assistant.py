"""Auto generated, do not edit."""
import datetime

import _zivid
import zivid._settings_converter as _settings_converter
from zivid._capture_assistant_suggest_settings_parameters_converter import (
    to_internal_capture_assistant_suggest_settings_parameters,
)


class SuggestSettingsParameters:
    """Class representing parameters to the Capture Assistant.

    An instance of this class is used as an argument to suggest_settings() in order
    to specify parameters and constraints to the capture assistant.
    """

    class AmbientLightFrequency:  # pylint: disable=too-few-public-methods
        """Optional choice for adapting to ambient light frequency.

        May help remove ripple artifacts caused by strong ambient light.
        """

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
            """Get list of allowed values.

            Returns:
                List of strings
            """
            return list(cls._valid_values.keys())

    def __init__(
        self,
        ambient_light_frequency=_zivid.capture_assistant.SuggestSettingsParameters()
        .AmbientLightFrequency()
        .value,
        max_capture_time=_zivid.capture_assistant.SuggestSettingsParameters()
        .MaxCaptureTime()
        .value,
    ):
        """Initialize a SuggestSettingsParameters.

        Pass this object to zivid.capture_assistant.suggest_settings() to run the Capture Assistant.

        Args:
            ambient_light_frequency: Optional ambient light adaptation ("none", "hz50" or "hz60")
            max_capture_time: A datetime.timedelta specifying maximum allowed HDR capture time

        Raises:
            TypeError: If one of the arguments is of the wrong type
        """
        if isinstance(
            ambient_light_frequency,
            _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.enum,
        ):
            self._ambient_light_frequency = ambient_light_frequency
        elif isinstance(ambient_light_frequency, (str,)):
            self._ambient_light_frequency = _convert_to_internal(
                ambient_light_frequency
            )
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(ambient_light_frequency)
                )
            )
        if isinstance(max_capture_time, (datetime.timedelta,)):
            self._max_capture_time = _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
                max_capture_time
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                    value_type=type(max_capture_time)
                )
            )

    @property
    def ambient_light_frequency(self):
        """Get the contained ambient light adaptation setting.

        Returns:
            A string, among the valid values of AmbientLightFrequency

        Raises:
            ValueError: If the object contains an invalid ambient light frequency setting
        """
        for (
            key,
            value,
        ) in (
            SuggestSettingsParameters.AmbientLightFrequency._valid_values.items()  # pylint: disable=protected-access
        ):
            if value == self._ambient_light_frequency:
                return key
        raise ValueError(
            "Unsupported value {value}".format(value=self._ambient_light_frequency)
        )

    @property
    def max_capture_time(self):
        """Get the contained max capture time setting.

        Returns:
            A datetime.timedelta object
        """
        return self._max_capture_time.value

    @ambient_light_frequency.setter
    def ambient_light_frequency(self, value):
        if isinstance(value, (str,)):
            self._ambient_light_frequency = _convert_to_internal(value)
        else:
            raise TypeError(
                "Unsupported type, expected: str, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @max_capture_time.setter
    def max_capture_time(self, value):
        if isinstance(value, (datetime.timedelta,)):
            self._max_capture_time = _zivid.capture_assistant.SuggestSettingsParameters.MaxCaptureTime(
                value
            )
        else:
            raise TypeError(
                "Unsupported type, expected: (datetime.timedelta,), got {value_type}".format(
                    value_type=type(value)
                )
            )

    def __eq__(self, other):
        if (
            self._ambient_light_frequency == other._ambient_light_frequency
            and self._max_capture_time == other._max_capture_time
        ):
            return True
        return False

    def __str__(self):
        return str(to_internal_capture_assistant_suggest_settings_parameters(self))


def _convert_to_internal(ambient_light_frequency):
    if isinstance(ambient_light_frequency, str):
        if (
            ambient_light_frequency
            not in SuggestSettingsParameters.AmbientLightFrequency._valid_values  # pylint: disable=protected-access
        ):
            raise TypeError(
                "Unsupported value {value}".format(value=ambient_light_frequency)
            )
        ambient_light_frequency = SuggestSettingsParameters.AmbientLightFrequency._valid_values[  # pylint: disable=protected-access
            ambient_light_frequency
        ]
        if not isinstance(
            ambient_light_frequency,
            _zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.enum,
        ):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(ambient_light_frequency))
            )
        return ambient_light_frequency
    raise TypeError(
        "Unsupported type: {value}".format(value=type(ambient_light_frequency))
    )


def suggest_settings(camera, suggest_settings_parameters):
    """Find settings for the current scene based on given parameters.

    The suggested settings returned from this function should be passed into
    camera.capture() to capture and retrieve the Frame containing a point cloud.

    Args:
        camera: A Camera instance
        suggest_settings_parameters: A SuggestSettingsParameters instance

    Returns:
        A Settings instance optimized for the current scene
    """
    internal_settings = _zivid.capture_assistant.suggest_settings(
        camera._Camera__impl,  # pylint: disable=protected-access
        to_internal_capture_assistant_suggest_settings_parameters(
            suggest_settings_parameters
        ),
    )
    return _settings_converter.to_settings(internal_settings)
