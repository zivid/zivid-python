"""Contains functions to convert between settings and internal settings."""
import _zivid
from zivid.settings import Settings


def to_settings(internal_settings):
    """Convert internal settings type to settings.

    Args:
        internal_settings: a internal settings object

    Returns:
        a settings object

    """

    def to_filters(internal_filters):
        def to_contrast(internal_contrast):
            return Settings.Filters.Contrast(
                enabled=internal_contrast.enabled.value,
                threshold=internal_contrast.threshold.value,
            )

        def to_outlier(internal_outlier):
            return Settings.Filters.Outlier(
                enabled=internal_outlier.enabled.value,
                threshold=internal_outlier.threshold.value,
            )

        def to_saturated(internal_saturated):
            return Settings.Filters.Saturated(enabled=internal_saturated.enabled.value)

        def to_reflection(internal_reflection):
            return Settings.Filters.Reflection(
                enabled=internal_reflection.enabled.value
            )

        def to_gaussian(internal_gaussian):
            return Settings.Filters.Gaussian(
                enabled=internal_gaussian.enabled.value,
                sigma=internal_gaussian.sigma.value,
            )

        return Settings.Filters(
            contrast=to_contrast(internal_filters.contrast),
            outlier=to_outlier(internal_filters.outlier),
            saturated=to_saturated(internal_filters.saturated),
            reflection=to_reflection(internal_filters.reflection),
            gaussian=to_gaussian(internal_filters.gaussian),
        )

    return Settings(
        bidirectional=internal_settings.bidirectional.value,
        blue_balance=internal_settings.bluebalance.value,
        brightness=internal_settings.brightness.value,
        exposure_time=internal_settings.exposuretime.value,
        filters=to_filters(internal_settings.filters),
        gain=internal_settings.gain.value,
        iris=internal_settings.iris.value,
        red_balance=internal_settings.redbalance.value,
    )


def to_internal_settings(settings):
    """Convert settings type to internal settings type.

    Args:
        settings: a settings object

    Returns:
        an internal settings object

    """

    def to_internal_bidirectional(bidirectional):
        return _zivid.Settings.Bidirectional(bidirectional)

    def to_internal_blue_balance(blue_balance):
        return _zivid.Settings.BlueBalance(blue_balance)

    def to_internal_brightness(brightness):
        return _zivid.Settings.Brightness(brightness)

    def to_internal_exposure_time(exposure_time):
        return _zivid.Settings.ExposureTime(exposure_time)

    def to_internal_gain(gain):
        return _zivid.Settings.Gain(gain)

    def to_internal_iris(iris):
        return _zivid.Settings.Iris(iris)

    def to_internal_red_balance(red_balance):
        return _zivid.Settings.RedBalance(red_balance)

    def to_internal_filters(filters):
        internal_filters = _zivid.Settings.Filters()
        internal_filters.contrast = to_internal_contrast(filters.contrast)
        internal_filters.outlier = to_internal_outlier(filters.outlier)
        internal_filters.saturated = to_internal_saturated(filters.saturated)
        internal_filters.reflection = to_internal_reflection(filters.reflection)
        internal_filters.gaussian = to_internal_gaussian(filters.gaussian)
        return internal_filters

    def to_internal_contrast(contrast):
        internal_contrast = _zivid.Settings.Filters.Contrast()
        internal_contrast.enabled = _zivid.Settings.Filters.Contrast.Enabled(
            contrast.enabled
        )
        internal_contrast.threshold = _zivid.Settings.Filters.Contrast.Threshold(
            contrast.threshold
        )
        return internal_contrast

    def to_internal_outlier(outlier):
        internal_outlier = _zivid.Settings.Filters.Outlier()
        internal_outlier.enabled = _zivid.Settings.Filters.Outlier.Enabled(
            outlier.enabled
        )
        internal_outlier.threshold = _zivid.Settings.Filters.Outlier.Threshold(
            outlier.threshold
        )
        return internal_outlier

    def to_internal_saturated(saturated):
        internal_saturated = _zivid.Settings.Filters.Saturated()
        internal_saturated.enabled = _zivid.Settings.Filters.Saturated.Enabled(
            saturated.enabled
        )
        return internal_saturated

    def to_internal_reflection(reflection):
        internal_reflection = _zivid.Settings.Filters.Reflection()
        internal_reflection.enabled = _zivid.Settings.Filters.Reflection.Enabled(
            reflection.enabled
        )
        return internal_reflection

    def to_internal_gaussian(gaussian):
        internal_gaussian = _zivid.Settings.Filters.Gaussian()
        internal_gaussian.enabled = _zivid.Settings.Filters.Gaussian.Enabled(
            gaussian.enabled
        )
        internal_gaussian.sigma = _zivid.Settings.Filters.Gaussian.Sigma(gaussian.sigma)
        return internal_gaussian

    internal_settings = _zivid.Settings()
    internal_settings.bidirectional = to_internal_bidirectional(settings.bidirectional)
    internal_settings.bluebalance = to_internal_blue_balance(settings.blue_balance)
    internal_settings.brightness = to_internal_brightness(settings.brightness)
    internal_settings.exposuretime = to_internal_exposure_time(settings.exposure_time)
    internal_settings.filters = to_internal_filters(settings.filters)
    internal_settings.gain = to_internal_gain(settings.gain)
    internal_settings.iris = to_internal_iris(settings.iris)
    internal_settings.redbalance = to_internal_red_balance(settings.red_balance)
    return internal_settings
