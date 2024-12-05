"""Auto generated, do not edit."""

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime
import collections.abc
import _zivid


class Settings2D:

    class Acquisition:

        def __init__(
            self,
            aperture=_zivid.Settings2D.Acquisition.Aperture().value,
            brightness=_zivid.Settings2D.Acquisition.Brightness().value,
            exposure_time=_zivid.Settings2D.Acquisition.ExposureTime().value,
            gain=_zivid.Settings2D.Acquisition.Gain().value,
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
                self._aperture = _zivid.Settings2D.Acquisition.Aperture(aperture)
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
                self._brightness = _zivid.Settings2D.Acquisition.Brightness(brightness)
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
                self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
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
                self._gain = _zivid.Settings2D.Acquisition.Gain(gain)
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
                self._aperture = _zivid.Settings2D.Acquisition.Aperture(value)
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
                self._brightness = _zivid.Settings2D.Acquisition.Brightness(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @exposure_time.setter
        def exposure_time(self, value):
            if isinstance(value, (datetime.timedelta,)) or value is None:
                self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(value)
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
                self._gain = _zivid.Settings2D.Acquisition.Gain(value)
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
            return str(_to_internal_settings2d_acquisition(self))

    class Processing:

        class Color:

            class Balance:

                def __init__(
                    self,
                    blue=_zivid.Settings2D.Processing.Color.Balance.Blue().value,
                    green=_zivid.Settings2D.Processing.Color.Balance.Green().value,
                    red=_zivid.Settings2D.Processing.Color.Balance.Red().value,
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
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
                            blue
                        )
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
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(
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
                        self._red = _zivid.Settings2D.Processing.Color.Balance.Red(red)
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
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
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
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(
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
                        self._red = _zivid.Settings2D.Processing.Color.Balance.Red(
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
                        self._blue == other._blue
                        and self._green == other._green
                        and self._red == other._red
                    ):
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings2d_processing_color_balance(self))

            class Experimental:

                class Mode:

                    automatic = "automatic"
                    toneMapping = "toneMapping"

                    _valid_values = {
                        "automatic": _zivid.Settings2D.Processing.Color.Experimental.Mode.automatic,
                        "toneMapping": _zivid.Settings2D.Processing.Color.Experimental.Mode.toneMapping,
                    }

                    @classmethod
                    def valid_values(cls):
                        return list(cls._valid_values.keys())

                def __init__(
                    self,
                    mode=_zivid.Settings2D.Processing.Color.Experimental.Mode().value,
                ):

                    if (
                        isinstance(
                            mode,
                            _zivid.Settings2D.Processing.Color.Experimental.Mode.enum,
                        )
                        or mode is None
                    ):
                        self._mode = (
                            _zivid.Settings2D.Processing.Color.Experimental.Mode(mode)
                        )
                    elif isinstance(mode, str):
                        self._mode = (
                            _zivid.Settings2D.Processing.Color.Experimental.Mode(
                                self.Mode._valid_values[mode]
                            )
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(
                                value_type=type(mode)
                            )
                        )

                @property
                def mode(self):
                    if self._mode.value is None:
                        return None
                    for key, internal_value in self.Mode._valid_values.items():
                        if internal_value == self._mode.value:
                            return key
                    raise ValueError(
                        "Unsupported value {value}".format(value=self._mode)
                    )

                @mode.setter
                def mode(self, value):
                    if isinstance(value, str):
                        self._mode = (
                            _zivid.Settings2D.Processing.Color.Experimental.Mode(
                                self.Mode._valid_values[value]
                            )
                        )
                    elif (
                        isinstance(
                            value,
                            _zivid.Settings2D.Processing.Color.Experimental.Mode.enum,
                        )
                        or value is None
                    ):
                        self._mode = (
                            _zivid.Settings2D.Processing.Color.Experimental.Mode(value)
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(
                                value_type=type(value)
                            )
                        )

                def __eq__(self, other):
                    if self._mode == other._mode:
                        return True
                    return False

                def __str__(self):
                    return str(
                        _to_internal_settings2d_processing_color_experimental(self)
                    )

            def __init__(
                self,
                gamma=_zivid.Settings2D.Processing.Color.Gamma().value,
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
                    self._gamma = _zivid.Settings2D.Processing.Color.Gamma(gamma)
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
                    self._gamma = _zivid.Settings2D.Processing.Color.Gamma(value)
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
                return str(_to_internal_settings2d_processing_color(self))

        def __init__(
            self,
            color=None,
        ):

            if color is None:
                color = self.Color()
            if not isinstance(color, self.Color):
                raise TypeError("Unsupported type: {value}".format(value=type(color)))
            self._color = color

        @property
        def color(self):
            return self._color

        @color.setter
        def color(self, value):
            if not isinstance(value, self.Color):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._color = value

        def __eq__(self, other):
            if self._color == other._color:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings2d_processing(self))

    class Sampling:

        class Color:

            grayscale = "grayscale"
            rgb = "rgb"

            _valid_values = {
                "grayscale": _zivid.Settings2D.Sampling.Color.grayscale,
                "rgb": _zivid.Settings2D.Sampling.Color.rgb,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        class Pixel:

            all = "all"
            blueSubsample2x2 = "blueSubsample2x2"
            blueSubsample4x4 = "blueSubsample4x4"
            by2x2 = "by2x2"
            by4x4 = "by4x4"
            redSubsample2x2 = "redSubsample2x2"
            redSubsample4x4 = "redSubsample4x4"

            _valid_values = {
                "all": _zivid.Settings2D.Sampling.Pixel.all,
                "blueSubsample2x2": _zivid.Settings2D.Sampling.Pixel.blueSubsample2x2,
                "blueSubsample4x4": _zivid.Settings2D.Sampling.Pixel.blueSubsample4x4,
                "by2x2": _zivid.Settings2D.Sampling.Pixel.by2x2,
                "by4x4": _zivid.Settings2D.Sampling.Pixel.by4x4,
                "redSubsample2x2": _zivid.Settings2D.Sampling.Pixel.redSubsample2x2,
                "redSubsample4x4": _zivid.Settings2D.Sampling.Pixel.redSubsample4x4,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            color=_zivid.Settings2D.Sampling.Color().value,
            pixel=_zivid.Settings2D.Sampling.Pixel().value,
        ):

            if (
                isinstance(color, _zivid.Settings2D.Sampling.Color.enum)
                or color is None
            ):
                self._color = _zivid.Settings2D.Sampling.Color(color)
            elif isinstance(color, str):
                self._color = _zivid.Settings2D.Sampling.Color(
                    self.Color._valid_values[color]
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(color)
                    )
                )

            if (
                isinstance(pixel, _zivid.Settings2D.Sampling.Pixel.enum)
                or pixel is None
            ):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(pixel)
            elif isinstance(pixel, str):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(
                    self.Pixel._valid_values[pixel]
                )
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(pixel)
                    )
                )

        @property
        def color(self):
            if self._color.value is None:
                return None
            for key, internal_value in self.Color._valid_values.items():
                if internal_value == self._color.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._color))

        @property
        def pixel(self):
            if self._pixel.value is None:
                return None
            for key, internal_value in self.Pixel._valid_values.items():
                if internal_value == self._pixel.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._pixel))

        @color.setter
        def color(self, value):
            if isinstance(value, str):
                self._color = _zivid.Settings2D.Sampling.Color(
                    self.Color._valid_values[value]
                )
            elif (
                isinstance(value, _zivid.Settings2D.Sampling.Color.enum)
                or value is None
            ):
                self._color = _zivid.Settings2D.Sampling.Color(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @pixel.setter
        def pixel(self, value):
            if isinstance(value, str):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(
                    self.Pixel._valid_values[value]
                )
            elif (
                isinstance(value, _zivid.Settings2D.Sampling.Pixel.enum)
                or value is None
            ):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        def __eq__(self, other):
            if self._color == other._color and self._pixel == other._pixel:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings2d_sampling(self))

    def __init__(
        self,
        acquisitions=None,
        processing=None,
        sampling=None,
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

        if processing is None:
            processing = self.Processing()
        if not isinstance(processing, self.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

        if sampling is None:
            sampling = self.Sampling()
        if not isinstance(sampling, self.Sampling):
            raise TypeError("Unsupported type: {value}".format(value=type(sampling)))
        self._sampling = sampling

    @property
    def acquisitions(self):
        return self._acquisitions

    @property
    def processing(self):
        return self._processing

    @property
    def sampling(self):
        return self._sampling

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

    @processing.setter
    def processing(self, value):
        if not isinstance(value, self.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    @sampling.setter
    def sampling(self, value):
        if not isinstance(value, self.Sampling):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._sampling = value

    @classmethod
    def load(cls, file_name):
        return _to_settings2d(_zivid.Settings2D(str(file_name)))

    def save(self, file_name):
        _to_internal_settings2d(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_settings2d(_zivid.Settings2D.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_settings2d(self).serialize()

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._processing == other._processing
            and self._sampling == other._sampling
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_settings2d(self))


def _to_settings2d_acquisition(internal_acquisition):
    return Settings2D.Acquisition(
        aperture=internal_acquisition.aperture.value,
        brightness=internal_acquisition.brightness.value,
        exposure_time=internal_acquisition.exposure_time.value,
        gain=internal_acquisition.gain.value,
    )


def _to_settings2d_processing_color_balance(internal_balance):
    return Settings2D.Processing.Color.Balance(
        blue=internal_balance.blue.value,
        green=internal_balance.green.value,
        red=internal_balance.red.value,
    )


def _to_settings2d_processing_color_experimental(internal_experimental):
    return Settings2D.Processing.Color.Experimental(
        mode=internal_experimental.mode.value,
    )


def _to_settings2d_processing_color(internal_color):
    return Settings2D.Processing.Color(
        balance=_to_settings2d_processing_color_balance(internal_color.balance),
        experimental=_to_settings2d_processing_color_experimental(
            internal_color.experimental
        ),
        gamma=internal_color.gamma.value,
    )


def _to_settings2d_processing(internal_processing):
    return Settings2D.Processing(
        color=_to_settings2d_processing_color(internal_processing.color),
    )


def _to_settings2d_sampling(internal_sampling):
    return Settings2D.Sampling(
        color=internal_sampling.color.value,
        pixel=internal_sampling.pixel.value,
    )


def _to_settings2d(internal_settings2d):
    return Settings2D(
        acquisitions=[
            _to_settings2d_acquisition(value)
            for value in internal_settings2d.acquisitions.value
        ],
        processing=_to_settings2d_processing(internal_settings2d.processing),
        sampling=_to_settings2d_sampling(internal_settings2d.sampling),
    )


def _to_internal_settings2d_acquisition(acquisition):
    internal_acquisition = _zivid.Settings2D.Acquisition()

    internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture(
        acquisition.aperture
    )
    internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness(
        acquisition.brightness
    )
    internal_acquisition.exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
        acquisition.exposure_time
    )
    internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain(acquisition.gain)

    return internal_acquisition


def _to_internal_settings2d_processing_color_balance(balance):
    internal_balance = _zivid.Settings2D.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
        balance.blue
    )
    internal_balance.green = _zivid.Settings2D.Processing.Color.Balance.Green(
        balance.green
    )
    internal_balance.red = _zivid.Settings2D.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def _to_internal_settings2d_processing_color_experimental(experimental):
    internal_experimental = _zivid.Settings2D.Processing.Color.Experimental()

    internal_experimental.mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(
        experimental._mode.value
    )

    return internal_experimental


def _to_internal_settings2d_processing_color(color):
    internal_color = _zivid.Settings2D.Processing.Color()

    internal_color.gamma = _zivid.Settings2D.Processing.Color.Gamma(color.gamma)

    internal_color.balance = _to_internal_settings2d_processing_color_balance(
        color.balance
    )
    internal_color.experimental = _to_internal_settings2d_processing_color_experimental(
        color.experimental
    )
    return internal_color


def _to_internal_settings2d_processing(processing):
    internal_processing = _zivid.Settings2D.Processing()

    internal_processing.color = _to_internal_settings2d_processing_color(
        processing.color
    )
    return internal_processing


def _to_internal_settings2d_sampling(sampling):
    internal_sampling = _zivid.Settings2D.Sampling()

    internal_sampling.color = _zivid.Settings2D.Sampling.Color(sampling._color.value)
    internal_sampling.pixel = _zivid.Settings2D.Sampling.Pixel(sampling._pixel.value)

    return internal_sampling


def _to_internal_settings2d(settings2d):
    internal_settings2d = _zivid.Settings2D()

    temp_acquisitions = _zivid.Settings2D.Acquisitions()
    for value in settings2d.acquisitions:
        temp_acquisitions.append(_to_internal_settings2d_acquisition(value))
    internal_settings2d.acquisitions = temp_acquisitions

    internal_settings2d.processing = _to_internal_settings2d_processing(
        settings2d.processing
    )
    internal_settings2d.sampling = _to_internal_settings2d_sampling(settings2d.sampling)
    return internal_settings2d
