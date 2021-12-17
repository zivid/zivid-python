"""Auto generated, do not edit."""
# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,too-many-branches,too-many-boolean-expressions
import datetime
import collections.abc
import _zivid


class Settings:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings.Acquisition.Aperture().value,
            brightness=_zivid.Settings.Acquisition.Brightness().value,
            exposure_time=_zivid.Settings.Acquisition.ExposureTime().value,
            gain=_zivid.Settings.Acquisition.Gain().value,
        ):

            if (
                isinstance(
                    aperture,
                    (
                        float,
                        int,
                    ),
                )
                or aperture is None
            ):
                self._aperture = _zivid.Settings.Acquisition.Aperture(aperture)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(aperture)
                    )
                )

            if (
                isinstance(
                    brightness,
                    (
                        float,
                        int,
                    ),
                )
                or brightness is None
            ):
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

            if (
                isinstance(
                    gain,
                    (
                        float,
                        int,
                    ),
                )
                or gain is None
            ):
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
            if (
                isinstance(
                    value,
                    (
                        float,
                        int,
                    ),
                )
                or value is None
            ):
                self._aperture = _zivid.Settings.Acquisition.Aperture(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @brightness.setter
        def brightness(self, value):
            if (
                isinstance(
                    value,
                    (
                        float,
                        int,
                    ),
                )
                or value is None
            ):
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
            if (
                isinstance(
                    value,
                    (
                        float,
                        int,
                    ),
                )
                or value is None
            ):
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
            return str(_to_internal_settings_acquisition(self))

    class Diagnostics:
        def __init__(
            self,
            enabled=_zivid.Settings.Diagnostics.Enabled().value,
        ):

            if isinstance(enabled, (bool,)) or enabled is None:
                self._enabled = _zivid.Settings.Diagnostics.Enabled(enabled)
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
                self._enabled = _zivid.Settings.Diagnostics.Enabled(value)
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
            return str(_to_internal_settings_diagnostics(self))

    class Experimental:
        class Engine:

            phase = "phase"
            stripe = "stripe"

            _valid_values = {
                "phase": _zivid.Settings.Experimental.Engine.phase,
                "stripe": _zivid.Settings.Experimental.Engine.stripe,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            engine=_zivid.Settings.Experimental.Engine().value,
        ):

            if (
                isinstance(engine, _zivid.Settings.Experimental.Engine.enum)
                or engine is None
            ):
                self._engine = _zivid.Settings.Experimental.Engine(engine)
            elif isinstance(engine, str):
                self._engine = _zivid.Settings.Experimental.Engine(
                    self.Engine._valid_values[engine]
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(engine)
                    )
                )

        @property
        def engine(self):
            if self._engine.value is None:
                return None
            for key, internal_value in self.Engine._valid_values.items():
                if internal_value == self._engine.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._engine))

        @engine.setter
        def engine(self, value):
            if isinstance(value, str):
                self._engine = _zivid.Settings.Experimental.Engine(
                    self.Engine._valid_values[value]
                )
            elif (
                isinstance(value, _zivid.Settings.Experimental.Engine.enum)
                or value is None
            ):
                self._engine = _zivid.Settings.Experimental.Engine(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._engine == other._engine:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_experimental(self))

    class Processing:
        class Color:
            class Balance:
                def __init__(
                    self,
                    blue=_zivid.Settings.Processing.Color.Balance.Blue().value,
                    green=_zivid.Settings.Processing.Color.Balance.Green().value,
                    red=_zivid.Settings.Processing.Color.Balance.Red().value,
                ):

                    if (
                        isinstance(
                            blue,
                            (
                                float,
                                int,
                            ),
                        )
                        or blue is None
                    ):
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue(blue)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(blue)
                            )
                        )

                    if (
                        isinstance(
                            green,
                            (
                                float,
                                int,
                            ),
                        )
                        or green is None
                    ):
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(
                            green
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(green)
                            )
                        )

                    if (
                        isinstance(
                            red,
                            (
                                float,
                                int,
                            ),
                        )
                        or red is None
                    ):
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
                    if (
                        isinstance(
                            value,
                            (
                                float,
                                int,
                            ),
                        )
                        or value is None
                    ):
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
                    if (
                        isinstance(
                            value,
                            (
                                float,
                                int,
                            ),
                        )
                        or value is None
                    ):
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
                    if (
                        isinstance(
                            value,
                            (
                                float,
                                int,
                            ),
                        )
                        or value is None
                    ):
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
                    return str(_to_internal_settings_processing_color_balance(self))

            class Experimental:
                class ToneMapping:
                    class Enabled:

                        always = "always"
                        hdrOnly = "hdrOnly"

                        _valid_values = {
                            "always": _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled.always,
                            "hdrOnly": _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled.hdrOnly,
                        }

                        @classmethod
                        def valid_values(cls):
                            return list(cls._valid_values.keys())

                    def __init__(
                        self,
                        enabled=_zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled().value,
                    ):

                        if (
                            isinstance(
                                enabled,
                                _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled.enum,
                            )
                            or enabled is None
                        ):
                            self._enabled = _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled(
                                enabled
                            )
                        elif isinstance(enabled, str):
                            self._enabled = _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled(
                                self.Enabled._valid_values[enabled]
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                    @property
                    def enabled(self):
                        if self._enabled.value is None:
                            return None
                        for key, internal_value in self.Enabled._valid_values.items():
                            if internal_value == self._enabled.value:
                                return key
                        raise ValueError(
                            "Unsupported value {value}".format(value=self._enabled)
                        )

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, str):
                            self._enabled = _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled(
                                self.Enabled._valid_values[value]
                            )
                        elif (
                            isinstance(
                                value,
                                _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled.enum,
                            )
                            or value is None
                        ):
                            self._enabled = _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled(
                                value
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled:
                            return True
                        return False

                    def __str__(self):
                        return str(
                            _to_internal_settings_processing_color_experimental_tone_mapping(
                                self
                            )
                        )

                def __init__(
                    self,
                    tone_mapping=None,
                ):

                    if tone_mapping is None:
                        tone_mapping = self.ToneMapping()
                    if not isinstance(tone_mapping, self.ToneMapping):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(tone_mapping))
                        )
                    self._tone_mapping = tone_mapping

                @property
                def tone_mapping(self):
                    return self._tone_mapping

                @tone_mapping.setter
                def tone_mapping(self, value):
                    if not isinstance(value, self.ToneMapping):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._tone_mapping = value

                def __eq__(self, other):
                    if self._tone_mapping == other._tone_mapping:
                        return True
                    return False

                def __str__(self):
                    return str(
                        _to_internal_settings_processing_color_experimental(self)
                    )

            def __init__(
                self,
                gamma=_zivid.Settings.Processing.Color.Gamma().value,
                balance=None,
                experimental=None,
            ):

                if (
                    isinstance(
                        gamma,
                        (
                            float,
                            int,
                        ),
                    )
                    or gamma is None
                ):
                    self._gamma = _zivid.Settings.Processing.Color.Gamma(gamma)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(gamma)
                        )
                    )

                if balance is None:
                    balance = self.Balance()
                if not isinstance(balance, self.Balance):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(balance))
                    )
                self._balance = balance

                if experimental is None:
                    experimental = self.Experimental()
                if not isinstance(experimental, self.Experimental):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(experimental))
                    )
                self._experimental = experimental

            @property
            def gamma(self):
                return self._gamma.value

            @property
            def balance(self):
                return self._balance

            @property
            def experimental(self):
                return self._experimental

            @gamma.setter
            def gamma(self, value):
                if (
                    isinstance(
                        value,
                        (
                            float,
                            int,
                        ),
                    )
                    or value is None
                ):
                    self._gamma = _zivid.Settings.Processing.Color.Gamma(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @balance.setter
            def balance(self, value):
                if not isinstance(value, self.Balance):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._balance = value

            @experimental.setter
            def experimental(self, value):
                if not isinstance(value, self.Experimental):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._experimental = value

            def __eq__(self, other):
                if (
                    self._gamma == other._gamma
                    and self._balance == other._balance
                    and self._experimental == other._experimental
                ):
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings_processing_color(self))

        class Filters:
            class Experimental:
                class ContrastDistortion:
                    class Correction:
                        def __init__(
                            self,
                            enabled=_zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled().value,
                            strength=_zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength().value,
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

                            if (
                                isinstance(
                                    strength,
                                    (
                                        float,
                                        int,
                                    ),
                                )
                                or strength is None
                            ):
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
                            if (
                                isinstance(
                                    value,
                                    (
                                        float,
                                        int,
                                    ),
                                )
                                or value is None
                            ):
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
                                _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
                                    self
                                )
                            )

                    class Removal:
                        def __init__(
                            self,
                            enabled=_zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled().value,
                            threshold=_zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold().value,
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
                                isinstance(
                                    threshold,
                                    (
                                        float,
                                        int,
                                    ),
                                )
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
                            if (
                                isinstance(
                                    value,
                                    (
                                        float,
                                        int,
                                    ),
                                )
                                or value is None
                            ):
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
                                _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
                                    self
                                )
                            )

                    def __init__(
                        self,
                        correction=None,
                        removal=None,
                    ):

                        if correction is None:
                            correction = self.Correction()
                        if not isinstance(correction, self.Correction):
                            raise TypeError(
                                "Unsupported type: {value}".format(
                                    value=type(correction)
                                )
                            )
                        self._correction = correction

                        if removal is None:
                            removal = self.Removal()
                        if not isinstance(removal, self.Removal):
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
                        if not isinstance(value, self.Correction):
                            raise TypeError(
                                "Unsupported type {value}".format(value=type(value))
                            )
                        self._correction = value

                    @removal.setter
                    def removal(self, value):
                        if not isinstance(value, self.Removal):
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
                            _to_internal_settings_processing_filters_experimental_contrast_distortion(
                                self
                            )
                        )

                def __init__(
                    self,
                    contrast_distortion=None,
                ):

                    if contrast_distortion is None:
                        contrast_distortion = self.ContrastDistortion()
                    if not isinstance(contrast_distortion, self.ContrastDistortion):
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
                    if not isinstance(value, self.ContrastDistortion):
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
                        _to_internal_settings_processing_filters_experimental(self)
                    )

            class Noise:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings.Processing.Filters.Noise.Removal.Enabled().value,
                        threshold=_zivid.Settings.Processing.Filters.Noise.Removal.Threshold().value,
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

                        if (
                            isinstance(
                                threshold,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or threshold is None
                        ):
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
                        if (
                            isinstance(
                                value,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or value is None
                        ):
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
                            _to_internal_settings_processing_filters_noise_removal(self)
                        )

                def __init__(
                    self,
                    removal=None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_noise(self))

            class Outlier:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings.Processing.Filters.Outlier.Removal.Enabled().value,
                        threshold=_zivid.Settings.Processing.Filters.Outlier.Removal.Threshold().value,
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

                        if (
                            isinstance(
                                threshold,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or threshold is None
                        ):
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
                        if (
                            isinstance(
                                value,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or value is None
                        ):
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
                            _to_internal_settings_processing_filters_outlier_removal(
                                self
                            )
                        )

                def __init__(
                    self,
                    removal=None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_outlier(self))

            class Reflection:
                class Removal:
                    def __init__(
                        self,
                        enabled=_zivid.Settings.Processing.Filters.Reflection.Removal.Enabled().value,
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
                            _to_internal_settings_processing_filters_reflection_removal(
                                self
                            )
                        )

                def __init__(
                    self,
                    removal=None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(removal))
                        )
                    self._removal = removal

                @property
                def removal(self):
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
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
                        _to_internal_settings_processing_filters_reflection(self)
                    )

            class Smoothing:
                class Gaussian:
                    def __init__(
                        self,
                        enabled=_zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled().value,
                        sigma=_zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma().value,
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

                        if (
                            isinstance(
                                sigma,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or sigma is None
                        ):
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
                        if (
                            isinstance(
                                value,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or value is None
                        ):
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
                            _to_internal_settings_processing_filters_smoothing_gaussian(
                                self
                            )
                        )

                def __init__(
                    self,
                    gaussian=None,
                ):

                    if gaussian is None:
                        gaussian = self.Gaussian()
                    if not isinstance(gaussian, self.Gaussian):
                        raise TypeError(
                            "Unsupported type: {value}".format(value=type(gaussian))
                        )
                    self._gaussian = gaussian

                @property
                def gaussian(self):
                    return self._gaussian

                @gaussian.setter
                def gaussian(self, value):
                    if not isinstance(value, self.Gaussian):
                        raise TypeError(
                            "Unsupported type {value}".format(value=type(value))
                        )
                    self._gaussian = value

                def __eq__(self, other):
                    if self._gaussian == other._gaussian:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_smoothing(self))

            def __init__(
                self,
                experimental=None,
                noise=None,
                outlier=None,
                reflection=None,
                smoothing=None,
            ):

                if experimental is None:
                    experimental = self.Experimental()
                if not isinstance(experimental, self.Experimental):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(experimental))
                    )
                self._experimental = experimental

                if noise is None:
                    noise = self.Noise()
                if not isinstance(noise, self.Noise):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(noise))
                    )
                self._noise = noise

                if outlier is None:
                    outlier = self.Outlier()
                if not isinstance(outlier, self.Outlier):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(outlier))
                    )
                self._outlier = outlier

                if reflection is None:
                    reflection = self.Reflection()
                if not isinstance(reflection, self.Reflection):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(reflection))
                    )
                self._reflection = reflection

                if smoothing is None:
                    smoothing = self.Smoothing()
                if not isinstance(smoothing, self.Smoothing):
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
                if not isinstance(value, self.Experimental):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._experimental = value

            @noise.setter
            def noise(self, value):
                if not isinstance(value, self.Noise):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._noise = value

            @outlier.setter
            def outlier(self, value):
                if not isinstance(value, self.Outlier):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._outlier = value

            @reflection.setter
            def reflection(self, value):
                if not isinstance(value, self.Reflection):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._reflection = value

            @smoothing.setter
            def smoothing(self, value):
                if not isinstance(value, self.Smoothing):
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
                return str(_to_internal_settings_processing_filters(self))

        def __init__(
            self,
            color=None,
            filters=None,
        ):

            if color is None:
                color = self.Color()
            if not isinstance(color, self.Color):
                raise TypeError("Unsupported type: {value}".format(value=type(color)))
            self._color = color

            if filters is None:
                filters = self.Filters()
            if not isinstance(filters, self.Filters):
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
            if not isinstance(value, self.Color):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._color = value

        @filters.setter
        def filters(self, value):
            if not isinstance(value, self.Filters):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._filters = value

        def __eq__(self, other):
            if self._color == other._color and self._filters == other._filters:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_processing(self))

    def __init__(
        self,
        acquisitions=None,
        diagnostics=None,
        experimental=None,
        processing=None,
    ):

        if acquisitions is None:
            self._acquisitions = []
        elif isinstance(acquisitions, (collections.abc.Iterable,)):
            self._acquisitions = []
            for item in acquisitions:
                if isinstance(item, self.Acquisition):
                    self._acquisitions.append(item)
                else:
                    raise TypeError(
                        "Unsupported type {item_type}".format(item_type=type(item))
                    )
        else:
            raise TypeError(
                "Unsupported type, expected: (collections.abc.Iterable,) or None, got {value_type}".format(
                    value_type=type(acquisitions)
                )
            )

        if diagnostics is None:
            diagnostics = self.Diagnostics()
        if not isinstance(diagnostics, self.Diagnostics):
            raise TypeError("Unsupported type: {value}".format(value=type(diagnostics)))
        self._diagnostics = diagnostics

        if experimental is None:
            experimental = self.Experimental()
        if not isinstance(experimental, self.Experimental):
            raise TypeError(
                "Unsupported type: {value}".format(value=type(experimental))
            )
        self._experimental = experimental

        if processing is None:
            processing = self.Processing()
        if not isinstance(processing, self.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

    @property
    def acquisitions(self):
        return self._acquisitions

    @property
    def diagnostics(self):
        return self._diagnostics

    @property
    def experimental(self):
        return self._experimental

    @property
    def processing(self):
        return self._processing

    @acquisitions.setter
    def acquisitions(self, value):
        if not isinstance(value, (collections.abc.Iterable,)):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._acquisitions = []
        for item in value:
            if isinstance(item, self.Acquisition):
                self._acquisitions.append(item)
            else:
                raise TypeError(
                    "Unsupported type {item_type}".format(item_type=type(item))
                )

    @diagnostics.setter
    def diagnostics(self, value):
        if not isinstance(value, self.Diagnostics):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._diagnostics = value

    @experimental.setter
    def experimental(self, value):
        if not isinstance(value, self.Experimental):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._experimental = value

    @processing.setter
    def processing(self, value):
        if not isinstance(value, self.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    @classmethod
    def load(cls, file_name):
        return _to_settings(_zivid.Settings(str(file_name)))

    def save(self, file_name):
        _to_internal_settings(self).save(str(file_name))

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._diagnostics == other._diagnostics
            and self._experimental == other._experimental
            and self._processing == other._processing
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_settings(self))


def _to_settings_acquisition(internal_acquisition):
    return Settings.Acquisition(
        aperture=internal_acquisition.aperture.value,
        brightness=internal_acquisition.brightness.value,
        exposure_time=internal_acquisition.exposure_time.value,
        gain=internal_acquisition.gain.value,
    )


def _to_settings_diagnostics(internal_diagnostics):
    return Settings.Diagnostics(
        enabled=internal_diagnostics.enabled.value,
    )


def _to_settings_experimental(internal_experimental):
    return Settings.Experimental(
        engine=internal_experimental.engine.value,
    )


def _to_settings_processing_color_balance(internal_balance):
    return Settings.Processing.Color.Balance(
        blue=internal_balance.blue.value,
        green=internal_balance.green.value,
        red=internal_balance.red.value,
    )


def _to_settings_processing_color_experimental_tone_mapping(internal_tone_mapping):
    return Settings.Processing.Color.Experimental.ToneMapping(
        enabled=internal_tone_mapping.enabled.value,
    )


def _to_settings_processing_color_experimental(internal_experimental):
    return Settings.Processing.Color.Experimental(
        tone_mapping=_to_settings_processing_color_experimental_tone_mapping(
            internal_experimental.tone_mapping
        ),
    )


def _to_settings_processing_color(internal_color):
    return Settings.Processing.Color(
        balance=_to_settings_processing_color_balance(internal_color.balance),
        experimental=_to_settings_processing_color_experimental(
            internal_color.experimental
        ),
        gamma=internal_color.gamma.value,
    )


def _to_settings_processing_filters_experimental_contrast_distortion_correction(
    internal_correction,
):
    return Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
        enabled=internal_correction.enabled.value,
        strength=internal_correction.strength.value,
    )


def _to_settings_processing_filters_experimental_contrast_distortion_removal(
    internal_removal,
):
    return Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_experimental_contrast_distortion(
    internal_contrast_distortion,
):
    return Settings.Processing.Filters.Experimental.ContrastDistortion(
        correction=_to_settings_processing_filters_experimental_contrast_distortion_correction(
            internal_contrast_distortion.correction
        ),
        removal=_to_settings_processing_filters_experimental_contrast_distortion_removal(
            internal_contrast_distortion.removal
        ),
    )


def _to_settings_processing_filters_experimental(internal_experimental):
    return Settings.Processing.Filters.Experimental(
        contrast_distortion=_to_settings_processing_filters_experimental_contrast_distortion(
            internal_experimental.contrast_distortion
        ),
    )


def _to_settings_processing_filters_noise_removal(internal_removal):
    return Settings.Processing.Filters.Noise.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_noise(internal_noise):
    return Settings.Processing.Filters.Noise(
        removal=_to_settings_processing_filters_noise_removal(internal_noise.removal),
    )


def _to_settings_processing_filters_outlier_removal(internal_removal):
    return Settings.Processing.Filters.Outlier.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_outlier(internal_outlier):
    return Settings.Processing.Filters.Outlier(
        removal=_to_settings_processing_filters_outlier_removal(
            internal_outlier.removal
        ),
    )


def _to_settings_processing_filters_reflection_removal(internal_removal):
    return Settings.Processing.Filters.Reflection.Removal(
        enabled=internal_removal.enabled.value,
    )


def _to_settings_processing_filters_reflection(internal_reflection):
    return Settings.Processing.Filters.Reflection(
        removal=_to_settings_processing_filters_reflection_removal(
            internal_reflection.removal
        ),
    )


def _to_settings_processing_filters_smoothing_gaussian(internal_gaussian):
    return Settings.Processing.Filters.Smoothing.Gaussian(
        enabled=internal_gaussian.enabled.value,
        sigma=internal_gaussian.sigma.value,
    )


def _to_settings_processing_filters_smoothing(internal_smoothing):
    return Settings.Processing.Filters.Smoothing(
        gaussian=_to_settings_processing_filters_smoothing_gaussian(
            internal_smoothing.gaussian
        ),
    )


def _to_settings_processing_filters(internal_filters):
    return Settings.Processing.Filters(
        experimental=_to_settings_processing_filters_experimental(
            internal_filters.experimental
        ),
        noise=_to_settings_processing_filters_noise(internal_filters.noise),
        outlier=_to_settings_processing_filters_outlier(internal_filters.outlier),
        reflection=_to_settings_processing_filters_reflection(
            internal_filters.reflection
        ),
        smoothing=_to_settings_processing_filters_smoothing(internal_filters.smoothing),
    )


def _to_settings_processing(internal_processing):
    return Settings.Processing(
        color=_to_settings_processing_color(internal_processing.color),
        filters=_to_settings_processing_filters(internal_processing.filters),
    )


def _to_settings(internal_settings):
    return Settings(
        acquisitions=[
            _to_settings_acquisition(value)
            for value in internal_settings.acquisitions.value
        ],
        diagnostics=_to_settings_diagnostics(internal_settings.diagnostics),
        experimental=_to_settings_experimental(internal_settings.experimental),
        processing=_to_settings_processing(internal_settings.processing),
    )


def _to_internal_settings_acquisition(acquisition):
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


def _to_internal_settings_diagnostics(diagnostics):
    internal_diagnostics = _zivid.Settings.Diagnostics()

    internal_diagnostics.enabled = _zivid.Settings.Diagnostics.Enabled(
        diagnostics.enabled
    )

    return internal_diagnostics


def _to_internal_settings_experimental(experimental):
    internal_experimental = _zivid.Settings.Experimental()

    internal_experimental.engine = _zivid.Settings.Experimental.Engine(
        experimental._engine.value
    )

    return internal_experimental


def _to_internal_settings_processing_color_balance(balance):
    internal_balance = _zivid.Settings.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings.Processing.Color.Balance.Blue(balance.blue)
    internal_balance.green = _zivid.Settings.Processing.Color.Balance.Green(
        balance.green
    )
    internal_balance.red = _zivid.Settings.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def _to_internal_settings_processing_color_experimental_tone_mapping(tone_mapping):
    internal_tone_mapping = _zivid.Settings.Processing.Color.Experimental.ToneMapping()

    internal_tone_mapping.enabled = (
        _zivid.Settings.Processing.Color.Experimental.ToneMapping.Enabled(
            tone_mapping._enabled.value
        )
    )

    return internal_tone_mapping


def _to_internal_settings_processing_color_experimental(experimental):
    internal_experimental = _zivid.Settings.Processing.Color.Experimental()

    internal_experimental.tone_mapping = (
        _to_internal_settings_processing_color_experimental_tone_mapping(
            experimental.tone_mapping
        )
    )
    return internal_experimental


def _to_internal_settings_processing_color(color):
    internal_color = _zivid.Settings.Processing.Color()

    internal_color.gamma = _zivid.Settings.Processing.Color.Gamma(color.gamma)

    internal_color.balance = _to_internal_settings_processing_color_balance(
        color.balance
    )
    internal_color.experimental = _to_internal_settings_processing_color_experimental(
        color.experimental
    )
    return internal_color


def _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
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


def _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
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


def _to_internal_settings_processing_filters_experimental_contrast_distortion(
    contrast_distortion,
):
    internal_contrast_distortion = (
        _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion()
    )

    internal_contrast_distortion.correction = _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
        contrast_distortion.correction
    )
    internal_contrast_distortion.removal = _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(
        contrast_distortion.removal
    )
    return internal_contrast_distortion


def _to_internal_settings_processing_filters_experimental(experimental):
    internal_experimental = _zivid.Settings.Processing.Filters.Experimental()

    internal_experimental.contrast_distortion = (
        _to_internal_settings_processing_filters_experimental_contrast_distortion(
            experimental.contrast_distortion
        )
    )
    return internal_experimental


def _to_internal_settings_processing_filters_noise_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Noise.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
        removal.enabled
    )
    internal_removal.threshold = (
        _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(removal.threshold)
    )

    return internal_removal


def _to_internal_settings_processing_filters_noise(noise):
    internal_noise = _zivid.Settings.Processing.Filters.Noise()

    internal_noise.removal = _to_internal_settings_processing_filters_noise_removal(
        noise.removal
    )
    return internal_noise


def _to_internal_settings_processing_filters_outlier_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Outlier.Removal()

    internal_removal.enabled = (
        _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(removal.enabled)
    )
    internal_removal.threshold = (
        _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(removal.threshold)
    )

    return internal_removal


def _to_internal_settings_processing_filters_outlier(outlier):
    internal_outlier = _zivid.Settings.Processing.Filters.Outlier()

    internal_outlier.removal = _to_internal_settings_processing_filters_outlier_removal(
        outlier.removal
    )
    return internal_outlier


def _to_internal_settings_processing_filters_reflection_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Reflection.Removal()

    internal_removal.enabled = (
        _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(removal.enabled)
    )

    return internal_removal


def _to_internal_settings_processing_filters_reflection(reflection):
    internal_reflection = _zivid.Settings.Processing.Filters.Reflection()

    internal_reflection.removal = (
        _to_internal_settings_processing_filters_reflection_removal(reflection.removal)
    )
    return internal_reflection


def _to_internal_settings_processing_filters_smoothing_gaussian(gaussian):
    internal_gaussian = _zivid.Settings.Processing.Filters.Smoothing.Gaussian()

    internal_gaussian.enabled = (
        _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(gaussian.enabled)
    )
    internal_gaussian.sigma = (
        _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(gaussian.sigma)
    )

    return internal_gaussian


def _to_internal_settings_processing_filters_smoothing(smoothing):
    internal_smoothing = _zivid.Settings.Processing.Filters.Smoothing()

    internal_smoothing.gaussian = (
        _to_internal_settings_processing_filters_smoothing_gaussian(smoothing.gaussian)
    )
    return internal_smoothing


def _to_internal_settings_processing_filters(filters):
    internal_filters = _zivid.Settings.Processing.Filters()

    internal_filters.experimental = (
        _to_internal_settings_processing_filters_experimental(filters.experimental)
    )
    internal_filters.noise = _to_internal_settings_processing_filters_noise(
        filters.noise
    )
    internal_filters.outlier = _to_internal_settings_processing_filters_outlier(
        filters.outlier
    )
    internal_filters.reflection = _to_internal_settings_processing_filters_reflection(
        filters.reflection
    )
    internal_filters.smoothing = _to_internal_settings_processing_filters_smoothing(
        filters.smoothing
    )
    return internal_filters


def _to_internal_settings_processing(processing):
    internal_processing = _zivid.Settings.Processing()

    internal_processing.color = _to_internal_settings_processing_color(processing.color)
    internal_processing.filters = _to_internal_settings_processing_filters(
        processing.filters
    )
    return internal_processing


def _to_internal_settings(settings):
    internal_settings = _zivid.Settings()

    temp_acquisitions = _zivid.Settings.Acquisitions()
    for value in settings.acquisitions:
        temp_acquisitions.append(_to_internal_settings_acquisition(value))
    internal_settings.acquisitions = temp_acquisitions

    internal_settings.diagnostics = _to_internal_settings_diagnostics(
        settings.diagnostics
    )
    internal_settings.experimental = _to_internal_settings_experimental(
        settings.experimental
    )
    internal_settings.processing = _to_internal_settings_processing(settings.processing)
    return internal_settings
