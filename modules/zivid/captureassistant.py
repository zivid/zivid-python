"""Contains capture assistant functions and classes."""
from enum import Enum

import _zivid
import zivid._settings_converter as _settings_converter


class AmbientLightFrequency(Enum):  # pylint: disable=too-few-public-methods
    """Ensure compatibility with the frequency of the ambient light in the scene."""

    hz50 = _zivid.captureassistant.AmbientLightFrequency.hz50
    hz60 = _zivid.captureassistant.AmbientLightFrequency.hz60
    none = _zivid.captureassistant.AmbientLightFrequency.none

    def __str__(self):
        return str(self.name)


class SuggestSettingsParameters:  # pylint: disable=too-few-public-methods
    """Input to the Capture Assistant algorithm.

    Used to specify a constraint on the total capture time for the settings suggested by the Capture Assistant,
    and optionally specify the ambient light frequency.
    The capture time constraint assumes a computer meeting Zivid's recommended minimum compute power.

    """

    def __init__(self, budget, frequency=None):
        """Initialize SuggestSettingsParameters.

        Args:
            budget: max capture time
            frequency: ambient light frequency

        """
        if frequency is None:
            self.__impl = _zivid.captureassistant.SuggestSettingsParameters(budget)
        else:
            self.__impl = _zivid.captureassistant.SuggestSettingsParameters(budget, frequency.value)

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
        return AmbientLightFrequency(self.__impl.ambientLightFrequency())

    def __str__(self):
        return self.__impl.to_string()


def suggest_settings(camera, suggest_settings_parameters):
    """Find settings for the current scene based on the suggest_settings_parameters.

    The suggested settings returned from this function should be passed into hdr.capture to perform the actual capture.

    Args:
        camera: reference to Camera instances
        suggest_settings_parameters: provides parameters (e.g., max capture time constraint)
                                     to the suggest_settings algorithm.

    Returns:
        List of Settings.

    """
    internal_settings = \
        _zivid.captureassistant.suggest_settings(
            camera._Camera__impl,    # pylint: disable=protected-access
            suggest_settings_parameters._SuggestSettingsParameters__impl)   # pylint: disable=protected-access
    return [_settings_converter.to_settings(internal) for internal in internal_settings]
