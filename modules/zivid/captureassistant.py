"""Contains capture assistant functions and classes."""
from enum import Enum
from collections import namedtuple

import _zivid
from zivid.settings import Settings
import zivid._settings_converter as _settings_converter


class AmbientLightFrequency(Enum):  # pylint: disable=too-few-public-methods
    AmbientLightFrequencyImpl = namedtuple('AmbientLightFrequency', ['value', 'to_string'] )

    hz50 = AmbientLightFrequencyImpl(_zivid.captureassistant.AmbientLightFrequency.hz50, \
                                     _zivid.captureassistant.AmbientLightFrequency.hz50.name)
    hz60 = AmbientLightFrequencyImpl(_zivid.captureassistant.AmbientLightFrequency.hz60, \
                                     _zivid.captureassistant.AmbientLightFrequency.hz60.name)
    none = AmbientLightFrequencyImpl(_zivid.captureassistant.AmbientLightFrequency.none, \
                                     _zivid.captureassistant.AmbientLightFrequency.none.name)
    def __str__(self):
        return self.value.to_string


def to_internal_ambient_light_frequency(ambient_light_frequency):
    to_internal_map = {
        AmbientLightFrequency.hz50 : _zivid.captureassistant.AmbientLightFrequency.hz50,
        AmbientLightFrequency.hz60 : _zivid.captureassistant.AmbientLightFrequency.hz60,
        AmbientLightFrequency.none : _zivid.captureassistant.AmbientLightFrequency.none,
    }
    return to_internal_map.get(ambient_light_frequency, "Invalid frequency.")


def to_ambient_light_frequency(internal_ambient_light_frequency):
    from_internal_map = {
        _zivid.captureassistant.AmbientLightFrequency.hz50 : AmbientLightFrequency.hz50,
        _zivid.captureassistant.AmbientLightFrequency.hz60 : AmbientLightFrequency.hz60,
        _zivid.captureassistant.AmbientLightFrequency.none : AmbientLightFrequency.none,
    }
    return from_internal_map.get(internal_ambient_light_frequency, "Invalid frequency.")


class SuggestSettingsParameters:  # pylint: disable=too-few-public-methods
    """Input to the Capture Assistant algorithm.

    Used to specify a constraint on the total capture time for the settings suggested by the Capture Assistant,
    and optionally specify the ambient light frequency.
    The capture time constraint assumes a computer meeting Zivid's recommended minimum compute power.

    """
    def __init__(self, budget, frequency=None):
        if frequency is None:
            self.__impl = _zivid.captureassistant.SuggestSettingsParameters(budget)
        else:
            self.__impl = _zivid.captureassistant.SuggestSettingsParameters(budget, \
                                                    to_internal_ambient_light_frequency(frequency))


    @property
    def max_capture_time(self):
        """Get capture-time budget.

        Returns:
            Instance of datetime.timedelta

        """
        return self.__impl.maxCaptureTime()


    @property
    def ambient_light_frequency(self):
        """Get ambient light frequency.

        Returns:
            Instance of AmbientLightFrequency

        """
        return to_ambient_light_frequency(self.__impl.ambientLightFrequency())


    def __str__(self):
        return self.__impl.to_string()


def suggest_settings(camera, suggest_settings_parameters):
    """Finds suggested settings for the current scene based on the suggest_settings_parameters.

    The suggested settings returned from this function should be passed into hdr.capture to perform the actual capture.

    Args:
        camera: reference to Camera instances
        suggest_settings_parameters: provides parameters (e.g., max capture time constraint)
                                     to the suggest_settings algorithm.

    Returns:
        List of Settings.

    """
    internal_settings = _zivid.captureassistant.suggest_settings(camera._Camera__impl, \
                                                    suggest_settings_parameters._SuggestSettingsParameters__impl)
    return [_settings_converter.to_settings(internal) for internal in internal_settings]
