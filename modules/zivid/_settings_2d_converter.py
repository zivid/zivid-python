"""Contains functions to convert between settings 2d and internal settings 2d."""
import _zivid
from zivid.settings_2d import Settings2D


def to_internal_settings_2d(settings_2d):
    """Convert settings 2d type to internal settings 2d type.

    Args:
        settings_2d: a settings 2d object

    Returns:
        an internal settings 2d object

    """

    def to_internal_brightness(brightness):
        return _zivid.Settings2D.Brightness(brightness)

    def to_internal_exposure_time(exposure_time):
        return _zivid.Settings2D.ExposureTime(exposure_time)

    def to_internal_gain(gain):
        return _zivid.Settings2D.Gain(gain)

    def to_internal_iris(iris):
        return _zivid.Settings2D.Iris(iris)

    internal_settings_2d = _zivid.Settings2D()
    internal_settings_2d.brightness = to_internal_brightness(settings_2d.brightness)
    internal_settings_2d.exposuretime = to_internal_exposure_time(
        settings_2d.exposure_time
    )
    internal_settings_2d.gain = to_internal_gain(settings_2d.gain)
    internal_settings_2d.iris = to_internal_iris(settings_2d.iris)
    return internal_settings_2d


def to_settings_2d(internal_settings_2d):
    """Convert internal settings 2d type to settings 2d.

    Args:
        internal_settings_2d: a internal settings 2d object

    Returns:
        a settings 2d object

    """

    return Settings2D(
        brightness=internal_settings_2d.brightness.value,
        exposure_time=internal_settings_2d.exposuretime.value,
        gain=internal_settings_2d.gain.value,
        iris=internal_settings_2d.iris.value,
    )
