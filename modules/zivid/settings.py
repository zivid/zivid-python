import _zivid
import zivid


class Settings:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings().Acquisition().Aperture().value,
            brightness=_zivid.Settings().Acquisition().Brightness().value,
            exposure_time=_zivid.Settings().Acquisition().ExposureTime().value,
            gain=_zivid.Settings().Acquisition().Gain().value,
        ):

            if aperture is not None:
                self._aperture = _zivid.Settings.Acquisition.Aperture(aperture)
            else:
                self._aperture = _zivid.Settings.Acquisition.Aperture()
            if brightness is not None:
                self._brightness = _zivid.Settings.Acquisition.Brightness(brightness)
            else:
                self._brightness = _zivid.Settings.Acquisition.Brightness()
            if exposure_time is not None:
                self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(
                    exposure_time
                )
            else:
                self._exposure_time = _zivid.Settings.Acquisition.ExposureTime()
            if gain is not None:
                self._gain = _zivid.Settings.Acquisition.Gain(gain)
            else:
                self._gain = _zivid.Settings.Acquisition.Gain()

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
            self._aperture = _zivid.Settings.Acquisition.Aperture(value)

        @brightness.setter
        def brightness(self, value):
            self._brightness = _zivid.Settings.Acquisition.Brightness(value)

        @exposure_time.setter
        def exposure_time(self, value):
            self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(value)

        @gain.setter
        def gain(self, value):
            self._gain = _zivid.Settings.Acquisition.Gain(value)

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
            return """Acquisition:
        aperture: {aperture}
        brightness: {brightness}
        exposure_time: {exposure_time}
        gain: {gain}
        """.format(
                aperture=self.aperture,
                brightness=self.brightness,
                exposure_time=self.exposure_time,
                gain=self.gain,
            )

    class Processing:
        class Color:
            class Balance:
                def __init__(
                    self,
                    blue=_zivid.Settings().Processing.Color.Balance().Blue().value,
                    green=_zivid.Settings().Processing.Color.Balance().Green().value,
                    red=_zivid.Settings().Processing.Color.Balance().Red().value,
                ):

                    if blue is not None:
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue(blue)
                    else:
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue()
                    if green is not None:
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(
                            green
                        )
                    else:
                        self._green = _zivid.Settings.Processing.Color.Balance.Green()
                    if red is not None:
                        self._red = _zivid.Settings.Processing.Color.Balance.Red(red)
                    else:
                        self._red = _zivid.Settings.Processing.Color.Balance.Red()

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
                    self._blue = _zivid.Settings.Processing.Color.Balance.Blue(value)

                @green.setter
                def green(self, value):
                    self._green = _zivid.Settings.Processing.Color.Balance.Green(value)

                @red.setter
                def red(self, value):
                    self._red = _zivid.Settings.Processing.Color.Balance.Red(value)

                def __eq__(self, other):
                    if (
                        self._blue == other._blue
                        and self._green == other._green
                        and self._red == other._red
                    ):
                        return True
                    return False

                def __str__(self):
                    return """Balance:
                blue: {blue}
                green: {green}
                red: {red}
                """.format(
                        blue=self.blue, green=self.green, red=self.red,
                    )

            def __init__(
                self, balance=None,
            ):

                if balance is None:
                    balance = zivid.Settings.Processing.Color.Balance()
                if not isinstance(balance, zivid.Settings.Processing.Color.Balance):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(balance))
                    )
                self._balance = balance

            @property
            def balance(self):
                return self._balance

            @balance.setter
            def balance(self, value):
                if not isinstance(value, zivid.Settings.Processing.Color.Balance):
                    raise TypeError(
                        "Unsupported type {value}".format(value=type(value))
                    )
                self._balance = value

            def __eq__(self, other):
                if self._balance == other._balance:
                    return True
                return False

            def __str__(self):
                return """Color:
            balance: {balance}
            """.format(
                    balance=self.balance,
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

                            if enabled is not None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
                                    enabled
                                )
                            else:
                                self._enabled = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled()
                                )
                            if strength is not None:
                                self._strength = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(
                                    strength
                                )
                            else:
                                self._strength = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength()
                                )

                        @property
                        def enabled(self):
                            return self._enabled.value

                        @property
                        def strength(self):
                            return self._strength.value

                        @enabled.setter
                        def enabled(self, value):
                            self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
                                value
                            )

                        @strength.setter
                        def strength(self, value):
                            self._strength = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(
                                value
                            )

                        def __eq__(self, other):
                            if (
                                self._enabled == other._enabled
                                and self._strength == other._strength
                            ):
                                return True
                            return False

                        def __str__(self):
                            return """Correction:
                        enabled: {enabled}
                        strength: {strength}
                        """.format(
                                enabled=self.enabled, strength=self.strength,
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

                            if enabled is not None:
                                self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                    enabled
                                )
                            else:
                                self._enabled = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled()
                                )
                            if threshold is not None:
                                self._threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
                                    threshold
                                )
                            else:
                                self._threshold = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold()
                                )

                        @property
                        def enabled(self):
                            return self._enabled.value

                        @property
                        def threshold(self):
                            return self._threshold.value

                        @enabled.setter
                        def enabled(self, value):
                            self._enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                value
                            )

                        @threshold.setter
                        def threshold(self, value):
                            self._threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
                                value
                            )

                        def __eq__(self, other):
                            if (
                                self._enabled == other._enabled
                                and self._threshold == other._threshold
                            ):
                                return True
                            return False

                        def __str__(self):
                            return """Removal:
                        enabled: {enabled}
                        threshold: {threshold}
                        """.format(
                                enabled=self.enabled, threshold=self.threshold,
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
                        return """ContrastDistortion:
                    correction: {correction}
                    removal: {removal}
                    """.format(
                            correction=self.correction, removal=self.removal,
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
                    return """Experimental:
                contrast_distortion: {contrast_distortion}
                """.format(
                        contrast_distortion=self.contrast_distortion,
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

                        if enabled is not None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
                                enabled
                            )
                        else:
                            self._enabled = (
                                _zivid.Settings.Processing.Filters.Noise.Removal.Enabled()
                            )
                        if threshold is not None:
                            self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(
                                threshold
                            )
                        else:
                            self._threshold = (
                                _zivid.Settings.Processing.Filters.Noise.Removal.Threshold()
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def threshold(self):
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(
                            value
                        )

                    @threshold.setter
                    def threshold(self, value):
                        self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(
                            value
                        )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._threshold == other._threshold
                        ):
                            return True
                        return False

                    def __str__(self):
                        return """Removal:
                    enabled: {enabled}
                    threshold: {threshold}
                    """.format(
                            enabled=self.enabled, threshold=self.threshold,
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
                    return """Noise:
                removal: {removal}
                """.format(
                        removal=self.removal,
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

                        if enabled is not None:
                            self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(
                                enabled
                            )
                        else:
                            self._enabled = (
                                _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled()
                            )
                        if threshold is not None:
                            self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(
                                threshold
                            )
                        else:
                            self._threshold = (
                                _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold()
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def threshold(self):
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(
                            value
                        )

                    @threshold.setter
                    def threshold(self, value):
                        self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(
                            value
                        )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._threshold == other._threshold
                        ):
                            return True
                        return False

                    def __str__(self):
                        return """Removal:
                    enabled: {enabled}
                    threshold: {threshold}
                    """.format(
                            enabled=self.enabled, threshold=self.threshold,
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
                    return """Outlier:
                removal: {removal}
                """.format(
                        removal=self.removal,
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

                        if enabled is not None:
                            self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(
                                enabled
                            )
                        else:
                            self._enabled = (
                                _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled()
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @enabled.setter
                    def enabled(self, value):
                        self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(
                            value
                        )

                    def __eq__(self, other):
                        if self._enabled == other._enabled:
                            return True
                        return False

                    def __str__(self):
                        return """Removal:
                    enabled: {enabled}
                    """.format(
                            enabled=self.enabled,
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
                    return """Reflection:
                removal: {removal}
                """.format(
                        removal=self.removal,
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

                        if enabled is not None:
                            self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(
                                enabled
                            )
                        else:
                            self._enabled = (
                                _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled()
                            )
                        if sigma is not None:
                            self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(
                                sigma
                            )
                        else:
                            self._sigma = (
                                _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma()
                            )

                    @property
                    def enabled(self):
                        return self._enabled.value

                    @property
                    def sigma(self):
                        return self._sigma.value

                    @enabled.setter
                    def enabled(self, value):
                        self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(
                            value
                        )

                    @sigma.setter
                    def sigma(self, value):
                        self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(
                            value
                        )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._sigma == other._sigma
                        ):
                            return True
                        return False

                    def __str__(self):
                        return """Gaussian:
                    enabled: {enabled}
                    sigma: {sigma}
                    """.format(
                            enabled=self.enabled, sigma=self.sigma,
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
                    return """Smoothing:
                gaussian: {gaussian}
                """.format(
                        gaussian=self.gaussian,
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
                return """Filters:
            experimental: {experimental}
            noise: {noise}
            outlier: {outlier}
            reflection: {reflection}
            smoothing: {smoothing}
            """.format(
                    experimental=self.experimental,
                    noise=self.noise,
                    outlier=self.outlier,
                    reflection=self.reflection,
                    smoothing=self.smoothing,
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
            return """Processing:
        color: {color}
        filters: {filters}
        """.format(
                color=self.color, filters=self.filters,
            )

    def __init__(
        self, acquisitions=None, processing=None,
    ):
        from collections.abc import Iterable

        if acquisitions is None:
            acquisitions = _zivid.Settings().Acquisitions().value
        if not isinstance(acquisitions, Iterable):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._acquisitions = _convert_to_acquistions(acquisitions)

        if processing is None:
            processing = zivid.Settings.Processing()
        if not isinstance(processing, zivid.Settings.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

    @property
    def acquisitions(self):
        return self._acquisitions

    @property
    def processing(self):
        return self._processing

    @processing.setter
    def processing(self, value):
        if not isinstance(value, zivid.Settings.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    @acquisitions.setter
    def acquisitions(self, value):
        from collections.abc import Iterable

        if not isinstance(value, Iterable):
            raise TypeError("Unsupported type: {value}".format(value=type(value)))
        self._acquisitions = _convert_to_acquistions(value)

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._processing == other._processing
        ):
            return True
        return False

    def __str__(self):
        return """Settings:
    acquisitions: {acquisitions}
    processing: {processing}
    """.format(
            acquisitions="\n".join([str(element) for element in self.acquisitions]),
            processing=self.processing,
        )


def _convert_to_acquistions(inputs):
    import zivid._settings_converter

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
