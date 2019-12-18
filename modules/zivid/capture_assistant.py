"""Contains capture assistant functions and classes."""
import _zivid
import zivid._settings_converter as _settings_converter
from zivid._make_enum_wrapper import _make_enum_wrapper


AmbientLightFrequency = _make_enum_wrapper(
    _zivid.capture_assistant.AmbientLightFrequency,
    "Ensure compatibility with the frequency of the ambient light in the scene.",
)


class SuggestSettingsParameters:  # pylint: disable=too-few-public-methods
    """Input to the Capture Assistant algorithm.

    Used to specify a constraint on the total capture time for the settings suggested by the Capture Assistant,
    and optionally specify the ambient light frequency.
    The capture time constraint assumes a computer meeting Zivid's recommended minimum compute power.

    """

    def __init__(self, max_capture_time, ambient_light_frequency=None):
        """Initialize SuggestSettingsParameters.

        Args:
            max_capture_time: an instance of datetime.timedelta
            ambient_light_frequency: a member of the enum zivid.capture_assistant.AmbientLightFrequency

        """
        if ambient_light_frequency is None:
            self.__impl = _zivid.capture_assistant.SuggestSettingsParameters(
                max_capture_time
            )
        else:
            self.__impl = _zivid.capture_assistant.SuggestSettingsParameters(
                max_capture_time,
                ambient_light_frequency._to_internal(),  # pylint: disable=protected-access
            )

    @property
    def max_capture_time(self):
        """Get max capture time.

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
        return AmbientLightFrequency(self.__impl.ambientLightFrequency().name)

    def __str__(self):
        return self.__impl.to_string()


def suggest_settings(camera, suggest_settings_parameters):
    """Find settings for the current scene based on the suggest_settings_parameters.

    The suggested settings returned from this function should be passed into hdr.capture to perform the actual capture.

    Args:
        camera: an instance of zivid.Camera
        suggest_settings_parameters: an instance of zivid.capture_assistant.SuggestSettingsParameters which provides
                                     parameters (e.g., max capture time constraint) to the suggest_settings algorithm.

    Returns:
        List of Settings.

    """
    internal_settings = _zivid.capture_assistant.suggest_settings(
        camera._Camera__impl,  # pylint: disable=protected-access
        suggest_settings_parameters._SuggestSettingsParameters__impl,  # pylint: disable=protected-access
    )
    return [_settings_converter.to_settings(internal) for internal in internal_settings]
