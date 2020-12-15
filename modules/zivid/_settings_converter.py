"""Auto generated, do not edit."""
import zivid
import _zivid


def to_settings_acquisition(internal_acquisition):
    return zivid.Settings.Acquisition(
        aperture=internal_acquisition.aperture.value,
        brightness=internal_acquisition.brightness.value,
        exposure_time=internal_acquisition.exposure_time.value,
        gain=internal_acquisition.gain.value,
    )


def to_settings_processing_color_balance(internal_balance):
    return zivid.Settings.Processing.Color.Balance(
        blue=internal_balance.blue.value,
        green=internal_balance.green.value,
        red=internal_balance.red.value,
    )


def to_settings_processing_color(internal_color):
    return zivid.Settings.Processing.Color(
        balance=to_settings_processing_color_balance(internal_color.balance),
        gamma=internal_color.gamma.value,
    )


def to_settings_processing_filters_experimental_contrast_distortion_correction(
    internal_correction,
):
    return zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
        enabled=internal_correction.enabled.value,
        strength=internal_correction.strength.value,
    )


def to_settings_processing_filters_experimental_contrast_distortion_removal(
    internal_removal,
):
    return zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def to_settings_processing_filters_experimental_contrast_distortion(
    internal_contrast_distortion,
):
    return zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(
        correction=to_settings_processing_filters_experimental_contrast_distortion_correction(
            internal_contrast_distortion.correction
        ),
        removal=to_settings_processing_filters_experimental_contrast_distortion_removal(
            internal_contrast_distortion.removal
        ),
    )


def to_settings_processing_filters_experimental(internal_experimental):
    return zivid.Settings.Processing.Filters.Experimental(
        contrast_distortion=to_settings_processing_filters_experimental_contrast_distortion(
            internal_experimental.contrast_distortion
        ),
    )


def to_settings_processing_filters_noise_removal(internal_removal):
    return zivid.Settings.Processing.Filters.Noise.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def to_settings_processing_filters_noise(internal_noise):
    return zivid.Settings.Processing.Filters.Noise(
        removal=to_settings_processing_filters_noise_removal(internal_noise.removal),
    )


def to_settings_processing_filters_outlier_removal(internal_removal):
    return zivid.Settings.Processing.Filters.Outlier.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def to_settings_processing_filters_outlier(internal_outlier):
    return zivid.Settings.Processing.Filters.Outlier(
        removal=to_settings_processing_filters_outlier_removal(
            internal_outlier.removal
        ),
    )


def to_settings_processing_filters_reflection_removal(internal_removal):
    return zivid.Settings.Processing.Filters.Reflection.Removal(
        enabled=internal_removal.enabled.value,
    )


def to_settings_processing_filters_reflection(internal_reflection):
    return zivid.Settings.Processing.Filters.Reflection(
        removal=to_settings_processing_filters_reflection_removal(
            internal_reflection.removal
        ),
    )


def to_settings_processing_filters_smoothing_gaussian(internal_gaussian):
    return zivid.Settings.Processing.Filters.Smoothing.Gaussian(
        enabled=internal_gaussian.enabled.value, sigma=internal_gaussian.sigma.value,
    )


def to_settings_processing_filters_smoothing(internal_smoothing):
    return zivid.Settings.Processing.Filters.Smoothing(
        gaussian=to_settings_processing_filters_smoothing_gaussian(
            internal_smoothing.gaussian
        ),
    )


def to_settings_processing_filters(internal_filters):
    return zivid.Settings.Processing.Filters(
        experimental=to_settings_processing_filters_experimental(
            internal_filters.experimental
        ),
        noise=to_settings_processing_filters_noise(internal_filters.noise),
        outlier=to_settings_processing_filters_outlier(internal_filters.outlier),
        reflection=to_settings_processing_filters_reflection(
            internal_filters.reflection
        ),
        smoothing=to_settings_processing_filters_smoothing(internal_filters.smoothing),
    )


def to_settings_processing(internal_processing):
    return zivid.Settings.Processing(
        color=to_settings_processing_color(internal_processing.color),
        filters=to_settings_processing_filters(internal_processing.filters),
    )


def to_settings(internal_settings):
    return zivid.Settings(
        processing=to_settings_processing(internal_settings.processing),
        acquisitions=[
            to_settings_acquisition(element)
            for element in internal_settings.acquisitions.value
        ],
    )


def to_internal_settings_acquisition(acquisition):
    internal_acquisition = _zivid.Settings.Acquisition()

    internal_acquisition.aperture = _zivid.Settings.Acquisition.Aperture(
        acquisition.aperture
    )
    internal_acquisition.brightness = _zivid.Settings.Acquisition.Brightness(
        acquisition.brightness
    )
    internal_acquisition.exposure_time = _zivid.Settings.Acquisition.ExposureTime(
        acquisition.exposure_time
    )
    internal_acquisition.gain = _zivid.Settings.Acquisition.Gain(acquisition.gain)

    return internal_acquisition


def to_internal_settings_processing_color_balance(balance):
    internal_balance = _zivid.Settings.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings.Processing.Color.Balance.Blue(balance.blue)
    internal_balance.green = _zivid.Settings.Processing.Color.Balance.Green(
        balance.green
    )
    internal_balance.red = _zivid.Settings.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def to_internal_settings_processing_color(color):
    internal_color = _zivid.Settings.Processing.Color()

    internal_color.gamma = _zivid.Settings.Processing.Color.Gamma(color.gamma)
    internal_color.balance = to_internal_settings_processing_color_balance(
        color.balance
    )
    return internal_color


def to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
    correction,
):
    internal_correction = (
        _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction()
    )

    internal_correction.enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
        correction.enabled
    )
    internal_correction.strength = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(
        correction.strength
    )

    return internal_correction


def to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
    removal,
):
    internal_removal = (
        _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal()
    )

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
        removal.enabled
    )
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
        removal.threshold
    )

    return internal_removal


def to_internal_settings_processing_filters_experimental_contrast_distortion(
    contrast_distortion,
):
    internal_contrast_distortion = (
        _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion()
    )

    internal_contrast_distortion.correction = to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
        contrast_distortion.correction
    )
    internal_contrast_distortion.removal = to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
        contrast_distortion.removal
    )
    return internal_contrast_distortion


def to_internal_settings_processing_filters_experimental(experimental):
    internal_experimental = _zivid.Settings.Processing.Filters.Experimental()

    internal_experimental.contrast_distortion = to_internal_settings_processing_filters_experimental_contrast_distortion(
        experimental.contrast_distortion
    )
    return internal_experimental


def to_internal_settings_processing_filters_noise_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Noise.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
        removal.enabled
    )
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(
        removal.threshold
    )

    return internal_removal


def to_internal_settings_processing_filters_noise(noise):
    internal_noise = _zivid.Settings.Processing.Filters.Noise()

    internal_noise.removal = to_internal_settings_processing_filters_noise_removal(
        noise.removal
    )
    return internal_noise


def to_internal_settings_processing_filters_outlier_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Outlier.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(
        removal.enabled
    )
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(
        removal.threshold
    )

    return internal_removal


def to_internal_settings_processing_filters_outlier(outlier):
    internal_outlier = _zivid.Settings.Processing.Filters.Outlier()

    internal_outlier.removal = to_internal_settings_processing_filters_outlier_removal(
        outlier.removal
    )
    return internal_outlier


def to_internal_settings_processing_filters_reflection_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Reflection.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(
        removal.enabled
    )

    return internal_removal


def to_internal_settings_processing_filters_reflection(reflection):
    internal_reflection = _zivid.Settings.Processing.Filters.Reflection()

    internal_reflection.removal = to_internal_settings_processing_filters_reflection_removal(
        reflection.removal
    )
    return internal_reflection


def to_internal_settings_processing_filters_smoothing_gaussian(gaussian):
    internal_gaussian = _zivid.Settings.Processing.Filters.Smoothing.Gaussian()

    internal_gaussian.enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(
        gaussian.enabled
    )
    internal_gaussian.sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(
        gaussian.sigma
    )

    return internal_gaussian


def to_internal_settings_processing_filters_smoothing(smoothing):
    internal_smoothing = _zivid.Settings.Processing.Filters.Smoothing()

    internal_smoothing.gaussian = to_internal_settings_processing_filters_smoothing_gaussian(
        smoothing.gaussian
    )
    return internal_smoothing


def to_internal_settings_processing_filters(filters):
    internal_filters = _zivid.Settings.Processing.Filters()

    internal_filters.experimental = to_internal_settings_processing_filters_experimental(
        filters.experimental
    )
    internal_filters.noise = to_internal_settings_processing_filters_noise(
        filters.noise
    )
    internal_filters.outlier = to_internal_settings_processing_filters_outlier(
        filters.outlier
    )
    internal_filters.reflection = to_internal_settings_processing_filters_reflection(
        filters.reflection
    )
    internal_filters.smoothing = to_internal_settings_processing_filters_smoothing(
        filters.smoothing
    )
    return internal_filters


def to_internal_settings_processing(processing):
    internal_processing = _zivid.Settings.Processing()

    internal_processing.color = to_internal_settings_processing_color(processing.color)
    internal_processing.filters = to_internal_settings_processing_filters(
        processing.filters
    )
    return internal_processing


def to_internal_settings(settings):
    internal_settings = _zivid.Settings()

    internal_settings.processing = to_internal_settings_processing(settings.processing)

    temp = _zivid.Settings().Acquisitions()
    for acq in settings.acquisitions:
        temp.append(to_internal_settings_acquisition(acq))
    internal_settings.acquisitions = temp

    return internal_settings
