"""Auto generated, do not edit."""
# pylint: disable=line-too-long,too-many-lines,too-many-arguments,missing-class-docstring,missing-function-docstring
import datetime
import collections.abc
import _zivid
import zivid
import zivid._settings_converter


class Settings:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings().Acquisition().Aperture().value,
            brightness=_zivid.Settings().Acquisition().Brightness().value,
            exposure_time=_zivid.Settings().Acquisition().ExposureTime().value,
            gain=_zivid.Settings().Acquisition().Gain().value,
        ):

            if isinstance(aperture, (float, int,)) or aperture is None:
                self._aperture = _zivid.Settings.Acquisition.Aperture(aperture)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(aperture)
                    )
                )
            if isinstance(brightness, (float, int,)) or brightness is None:
                self._brightness = _zivid.Settings.Acquisition.Brightness(brightness)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(brightness)
                    )
                )
            if (
                isinstance(exposure_time, (datetime.timedelta,))
                or exposure_time is None
            ):
                self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(
                    exposure_time
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: (datetime.timedelta,) or None, got {value_type}".format(
                        value_type=type(exposure_time)
                    )
                )
            if isinstance(gain, (float, int,)) or gain is None:
                self._gain = _zivid.Settings.Acquisition.Gain(gain)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(gain)
                    )
                )

        @property
        def aperture(self):
            return self._aperture.value

        @property
        def brightness(self):
            return self._brightness.value

        @property
        def exposure_time(self):
            return self._exposure_time.value

        @property
        def gain(self):
            return self._gain.value

        @aperture.setter
        def aperture(self, value):
            if isinstance(value, (float, int,)) or value is None:
                self._aperture = _zivid.Settings.Acquisition.Aperture(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @brightness.setter
        def brightness(self, value):
            if isinstance(value, (float, int,)) or value is None:
                self._brightness = _zivid.Settings.Acquisition.Brightness(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @exposure_time.setter
        def exposure_time(self, value):
            if isinstance(value, (datetime.timedelta,)) or value is None:
                self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: datetime.timedelta or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @gain.setter
        def gain(self, value):
            if isinstance(value, (float, int,)) or value is None:
                self._gain = _zivid.Settings.Acquisition.Gain(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if (
                self._aperture == other._aperture
                and self._brightness == other._brightness
                and self._exposure_time == other._exposure_time
                and self._gain == other._gain
            ):
                return True
            return False

        def __str__(self):
            return str(zivid._settings_converter.to_internal_settings_acquisition(self))

    class Processing:
        class Color:
            class Balance:
                def __init__(
                    self,
                    blue=_zivid.Settings().Processing.Color.Balance().Blue().value,
                    green=_zivid.Settings().Processing.Color.Balance().Green().value,
                    red=_zivid.Settings().Processing.Color.Balance().Red().value,
                ):

                    if isinstance(blue, (float, int,)) or blue is None:
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue(blue)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(blue)
                            )
                        )
                    if isinstance(green, (float, int,)) or green is None:
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(
                            green
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(green)
                            )
                        )
                    if isinstance(red, (float, int,)) or red is None:
                        self._red = _zivid.Settings.Processing.Color.Balance.Red(red)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(red)
                            )
                        )

                @property
                def blue(self):
                    return self._blue.value

                @property
                def green(self):
                    return self._green.value

                @property
                def red(self):
                    return self._red.value

                @blue.setter
                def blue(self, value):
                    if isinstance(value, (float, int,)) or value is None:
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue(
                            value
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                value_type=type(value)
                            )
                        )

                @green.setter
                def green(self, value):
                    if isinstance(value, (float, int,)) or value is None:
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(
                            value
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                value_type=type(value)
                            )
                        )

                @red.setter
                def red(self, value):
                    if isinstance(value, (float, int,)) or value is None:
                        self._red = _zivid.Settings.Processing.Color.Balance.Red(value)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                value_type=type(value)
                            )
                        )

                def __eq__(self, other):
                    if (
                        self._blue == other._blue
                        and self._green == other._green
                        and self._red == other._red
                    ):
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_color_balance(
                            self
                        )
                    )

            def __init__(
                self,
                gamma=_zivid.Settings().Processing.Color().Gamma().value,
                balance=None,
            ):

                if isinstance(gamma, (float, int,)) or gamma is None:
                    self._gamma = _zivid.Settings.Processing.Color.Gamma(gamma)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(gamma)
                        )
                    )
                if balance is None:
                    balance = zivid.Settings.Processing.Color.Balance()
                if not isinstance(balance, zivid.Settings.Processing.Color.Balance):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(balance))
                    )
                self._balance = balance

            @property
            def gamma(self):
                return self._gamma.value

            @property
            def balance(self):
                return self._balance

            @gamma.setter
            def gamma(self, value):
                if isinstance(value, (float, int,)) or value is None:
                    self._gamma = _zivid.Settings.Processing.Color.Gamma(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @balance.setter
            def balance(self, value):
                if not isinstance(value, zivid.Settings.Processing.Color.Balance):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._balance = value

            def __eq__(self, other):
                if self._gamma == other._gamma and self._balance == other._balance:
                    return True
                return False

            def __str__(self):
                return str(
                    zivid._settings_converter.to_internal_settings_processing_color(
                        self
                    )
                )

        class Filters:
            class Experimental:
                class ContrastDistortion:
                    class Correction:
                        def __init__(
                            self,
                            enabled=_zivid.Settings()
                            .Processing.Filters.Experimental.ContrastDistortion.Correction()
                            .Enabled()
                            .value,
                            strength=_zivid.Settings()
                            .Processing.Filters.Experimental.ContrastDistortion.Correction()
                            .Strength()
                            .value,
                        ):

                            if isinstance(enabled, (bool,)) or enabled is None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
                                    enabled
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                        value_type=type(enabled)
                                    )
                                )
                            if isinstance(strength, (float, int,)) or strength is None:
                                self._strength = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(
                                    strength
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                        value_type=type(strength)
                                    )
                                )

                        @property
                        def enabled(self):
                            return self._enabled.value

                        @property
                        def strength(self):
                            return self._strength.value

                        @enabled.setter
                        def enabled(self, value):
                            if isinstance(value, (bool,)) or value is None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
                                    value
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: bool or None, got {value_type}".format(
                                        value_type=type(value)
                                    )
                                )

                        @strength.setter
                        def strength(self, value):
                            if isinstance(value, (float, int,)) or value is None:
                                self._strength = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(
                                    value
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                        value_type=type(value)
                                    )
                                )

                        def __eq__(self, other):
                            if (
                                self._enabled == other._enabled
                                and self._strength == other._strength
                            ):
                                return True
                            return False

                        def __str__(self):
                            return str(
                                zivid._settings_converter.to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
                                    self
                                )
                            )

                    class Removal:
                        def __init__(
                            self,
                            enabled=_zivid.Settings()
                            .Processing.Filters.Experimental.ContrastDistortion.Removal()
                            .Enabled()
                            .value,
                            threshold=_zivid.Settings()
                            .Processing.Filters.Experimental.ContrastDistortion.Removal()
                            .Threshold()
                            .value,
                        ):

                            if isinstance(enabled, (bool,)) or enabled is None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                    enabled
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                        value_type=type(enabled)
                                    )
                                )
                            if (
                                isinstance(threshold, (float, int,))
                                or threshold is None
                            ):
                                self._threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
                                    threshold
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                        value_type=type(threshold)
                                    )
                                )

                        @property
                        def enabled(self):
                            return self._enabled.value

                        @property
                        def threshold(self):
                            return self._threshold.value

                        @enabled.setter
                        def enabled(self, value):
                            if isinstance(value, (bool,)) or value is None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                    value
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: bool or None, got {value_type}".format(
                                        value_type=type(value)
                                    )
                                )

                        @threshold.setter
                        def threshold(self, value):
                            if isinstance(value, (float, int,)) or value is None:
                                self._threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
                                    value
                                )
                            else:
                                raise TypeError(
                                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                        value_type=type(value)
                                    )
                                )

                        def __eq__(self, other):
                            if (
                                self._enabled == other._enabled
                                and self._threshold == other._threshold
                            ):
                                return True
                            return False

                        def __str__(self):
                            return str(
                                zivid._settings_converter.to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
                                    self
                                )
                            )

                    def __init__(
                        self, correction=None, removal=None,
                    ):

                        if correction is None:
                            correction = (
                                zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction()
                            )
                        if not isinstance(
                            correction,
                            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction,
                        ):
                            raise TypeError(
                                "Unsupported type: {value}".format(
                                    value=type(correction)
                                )
                            )
                        self._correction = correction
                        if removal is None:
                            removal = (
                                zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal()
                            )
                        if not isinstance(
                            removal,
                            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal,
                        ):
                            raise TypeError(
                                "Unsupported type: {value}".format(value=type(removal))
                            )
                        self._removal = removal

                    @property
                    def correction(self):
                        return self._correction

                    @property
                    def removal(self):
                        return self._removal

                    @correction.setter
                    def correction(self, value):
                        if not isinstance(
                            value,
                            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction,
                        ):
                            raise TypeError(
                                "Unsupported type {value}".format(value=type(value))
                            )
                        self._correction = value

                    @removal.setter
                    def removal(self, value):
                        if not isinstance(
                            value,
                            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal,
                        ):
                            raise TypeError(
                                "Unsupported type {value}".format(value=type(value))
                            )
                        self._removal = value

                    def __eq__(self, other):
                        if (
                            self._correction == other._correction
                            and self._removal == other._removal
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(
                            zivid._settings_converter.to_internal_settings_processing_filters_experimental_contrast_distortion(
                                self
                            )
                        )

                def __init__(
                    self, contrast_distortion=None,
                ):

                    if contrast_distortion is None:
                        contrast_distortion = (
                            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion()
                        )
                    if not isinstance(
                        contrast_distortion,
                        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion,
                    ):
                        raise TypeError(
                            "Unsupported type: {value}".format(
                                value=type(contrast_distortion)
                            )
                        )
                    self._contrast_distortion = contrast_distortion

                @property
                def contrast_distortion(self):
                    return self._contrast_distortion

                @contrast_distortion.setter
                def contrast_distortion(self, value):
                    if not isinstance(
                        value,
                        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion,
                    ):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._contrast_distortion = value

                def __eq__(self, other):
                    if self._contrast_distortion == other._contrast_distortion:
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_filters_experimental(
                            self
                        )
                    )

            class Noise:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings()
                        .Processing.Filters.Noise.Removal()
                        .Enabled()
                        .value,
                        threshold=_zivid.Settings()
                        .Processing.Filters.Noise.Removal()
                        .Threshold()
                        .value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
                                enabled
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )
                        if isinstance(threshold, (float, int,)) or threshold is None:
                            self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(
                                threshold
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(threshold)
                                )
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def threshold(self):
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @threshold.setter
                    def threshold(self, value):
                        if isinstance(value, (float, int,)) or value is None:
                            self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._threshold == other._threshold
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(
                            zivid._settings_converter.to_internal_settings_processing_filters_noise_removal(
                                self
                            )
                        )

                def __init__(
                    self, removal=None,
                ):

                    if removal is None:
                        removal = zivid.Settings.Processing.Filters.Noise.Removal()
                    if not isinstance(
                        removal, zivid.Settings.Processing.Filters.Noise.Removal
                    ):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(
                        value, zivid.Settings.Processing.Filters.Noise.Removal
                    ):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_filters_noise(
                            self
                        )
                    )

            class Outlier:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings()
                        .Processing.Filters.Outlier.Removal()
                        .Enabled()
                        .value,
                        threshold=_zivid.Settings()
                        .Processing.Filters.Outlier.Removal()
                        .Threshold()
                        .value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(
                                enabled
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )
                        if isinstance(threshold, (float, int,)) or threshold is None:
                            self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(
                                threshold
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(threshold)
                                )
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def threshold(self):
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @threshold.setter
                    def threshold(self, value):
                        if isinstance(value, (float, int,)) or value is None:
                            self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._threshold == other._threshold
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(
                            zivid._settings_converter.to_internal_settings_processing_filters_outlier_removal(
                                self
                            )
                        )

                def __init__(
                    self, removal=None,
                ):

                    if removal is None:
                        removal = zivid.Settings.Processing.Filters.Outlier.Removal()
                    if not isinstance(
                        removal, zivid.Settings.Processing.Filters.Outlier.Removal
                    ):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(
                        value, zivid.Settings.Processing.Filters.Outlier.Removal
                    ):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_filters_outlier(
                            self
                        )
                    )

            class Reflection:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings()
                        .Processing.Filters.Reflection.Removal()
                        .Enabled()
                        .value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(
                                enabled
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled:
                            return True
                        return False

                    def __str__(self):
                        return str(
                            zivid._settings_converter.to_internal_settings_processing_filters_reflection_removal(
                                self
                            )
                        )

                def __init__(
                    self, removal=None,
                ):

                    if removal is None:
                        removal = zivid.Settings.Processing.Filters.Reflection.Removal()
                    if not isinstance(
                        removal, zivid.Settings.Processing.Filters.Reflection.Removal
                    ):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(
                        value, zivid.Settings.Processing.Filters.Reflection.Removal
                    ):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_filters_reflection(
                            self
                        )
                    )

            class Smoothing:
                class Gaussian:
                    def __init__(
                        self,
                        enabled=_zivid.Settings()
                        .Processing.Filters.Smoothing.Gaussian()
                        .Enabled()
                        .value,
                        sigma=_zivid.Settings()
                        .Processing.Filters.Smoothing.Gaussian()
                        .Sigma()
                        .value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(
                                enabled
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )
                        if isinstance(sigma, (float, int,)) or sigma is None:
                            self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(
                                sigma
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(sigma)
                                )
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def sigma(self):
                        return self._sigma.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @sigma.setter
                    def sigma(self, value):
                        if isinstance(value, (float, int,)) or value is None:
                            self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._sigma == other._sigma
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(
                            zivid._settings_converter.to_internal_settings_processing_filters_smoothing_gaussian(
                                self
                            )
                        )

                def __init__(
                    self, gaussian=None,
                ):

                    if gaussian is None:
                        gaussian = (
                            zivid.Settings.Processing.Filters.Smoothing.Gaussian()
                        )
                    if not isinstance(
                        gaussian, zivid.Settings.Processing.Filters.Smoothing.Gaussian
                    ):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(gaussian))
                        )
                    self._gaussian = gaussian

                @property
                def gaussian(self):
                    return self._gaussian

                @gaussian.setter
                def gaussian(self, value):
                    if not isinstance(
                        value, zivid.Settings.Processing.Filters.Smoothing.Gaussian
                    ):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._gaussian = value

                def __eq__(self, other):
                    if self._gaussian == other._gaussian:
                        return True
                    return False

                def __str__(self):
                    return str(
                        zivid._settings_converter.to_internal_settings_processing_filters_smoothing(
                            self
                        )
                    )

            def __init__(
                self,
                experimental=None,
                noise=None,
                outlier=None,
                reflection=None,
                smoothing=None,
            ):

                if experimental is None:
                    experimental = zivid.Settings.Processing.Filters.Experimental()
                if not isinstance(
                    experimental, zivid.Settings.Processing.Filters.Experimental
                ):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(experimental))
                    )
                self._experimental = experimental
                if noise is None:
                    noise = zivid.Settings.Processing.Filters.Noise()
                if not isinstance(noise, zivid.Settings.Processing.Filters.Noise):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(noise))
                    )
                self._noise = noise
                if outlier is None:
                    outlier = zivid.Settings.Processing.Filters.Outlier()
                if not isinstance(outlier, zivid.Settings.Processing.Filters.Outlier):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(outlier))
                    )
                self._outlier = outlier
                if reflection is None:
                    reflection = zivid.Settings.Processing.Filters.Reflection()
                if not isinstance(
                    reflection, zivid.Settings.Processing.Filters.Reflection
                ):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(reflection))
                    )
                self._reflection = reflection
                if smoothing is None:
                    smoothing = zivid.Settings.Processing.Filters.Smoothing()
                if not isinstance(
                    smoothing, zivid.Settings.Processing.Filters.Smoothing
                ):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(smoothing))
                    )
                self._smoothing = smoothing

            @property
            def experimental(self):
                return self._experimental

            @property
            def noise(self):
                return self._noise

            @property
            def outlier(self):
                return self._outlier

            @property
            def reflection(self):
                return self._reflection

            @property
            def smoothing(self):
                return self._smoothing

            @experimental.setter
            def experimental(self, value):
                if not isinstance(
                    value, zivid.Settings.Processing.Filters.Experimental
                ):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._experimental = value

            @noise.setter
            def noise(self, value):
                if not isinstance(value, zivid.Settings.Processing.Filters.Noise):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._noise = value

            @outlier.setter
            def outlier(self, value):
                if not isinstance(value, zivid.Settings.Processing.Filters.Outlier):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._outlier = value

            @reflection.setter
            def reflection(self, value):
                if not isinstance(value, zivid.Settings.Processing.Filters.Reflection):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._reflection = value

            @smoothing.setter
            def smoothing(self, value):
                if not isinstance(value, zivid.Settings.Processing.Filters.Smoothing):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._smoothing = value

            def __eq__(self, other):
                if (
                    self._experimental == other._experimental
                    and self._noise == other._noise
                    and self._outlier == other._outlier
                    and self._reflection == other._reflection
                    and self._smoothing == other._smoothing
                ):
                    return True
                return False

            def __str__(self):
                return str(
                    zivid._settings_converter.to_internal_settings_processing_filters(
                        self
                    )
                )

        def __init__(
            self, color=None, filters=None,
        ):

            if color is None:
                color = zivid.Settings.Processing.Color()
            if not isinstance(color, zivid.Settings.Processing.Color):
                raise TypeError("Unsupported type: {value}".format(value=type(color)))
            self._color = color
            if filters is None:
                filters = zivid.Settings.Processing.Filters()
            if not isinstance(filters, zivid.Settings.Processing.Filters):
                raise TypeError("Unsupported type: {value}".format(value=type(filters)))
            self._filters = filters

        @property
        def color(self):
            return self._color

        @property
        def filters(self):
            return self._filters

        @color.setter
        def color(self, value):
            if not isinstance(value, zivid.Settings.Processing.Color):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._color = value

        @filters.setter
        def filters(self, value):
            if not isinstance(value, zivid.Settings.Processing.Filters):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._filters = value

        def __eq__(self, other):
            if self._color == other._color and self._filters == other._filters:
                return True
            return False

        def __str__(self):
            return str(zivid._settings_converter.to_internal_settings_processing(self))

    def __str__(self):
        return str(zivid._settings_converter.to_internal_settings(self))

    @property
    def processing(self):
        return self._processing

    @processing.setter
    def processing(self, value):
        if not isinstance(value, zivid.Settings.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    def __init__(
        self, acquisitions=None, processing=None,
    ):
        if acquisitions is None:
            acquisitions = _zivid.Settings().Acquisitions().value
        if not isinstance(acquisitions, collections.abc.Iterable):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(acquisitions))
            )
        self._acquisitions = _convert_to_acquistions(acquisitions)

        if processing is None:
            processing = zivid.Settings.Processing()
        if not isinstance(processing, zivid.Settings.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._processing == other._processing
        ):
            return True
        return False

    @property
    def acquisitions(self):
        return self._acquisitions

    @acquisitions.setter
    def acquisitions(self, value):
        if not isinstance(value, collections.abc.Iterable):
            raise TypeError("Unsupported type: {value}".format(value=type(value)))
        self._acquisitions = _convert_to_acquistions(value)


def _convert_to_acquistions(inputs):
    temp = []
    for acquisition_element in inputs:
        if isinstance(acquisition_element, Settings.Acquisition):
            temp.append(acquisition_element)
        else:
            raise TypeError(
                "Unsupported type {type_of_acquisition_element}".format(
                    type_of_acquisition_element=type(acquisition_element)
                )
            )
    return temp
