import _zivid
import zivid


class Settings2D:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings2D().Acquisition().Aperture().value,
            brightness=_zivid.Settings2D().Acquisition().Brightness().value,
            exposure_time=_zivid.Settings2D().Acquisition().ExposureTime().value,
            gain=_zivid.Settings2D().Acquisition().Gain().value,
        ):

            if aperture is not None:
                self._aperture = _zivid.Settings2D.Acquisition.Aperture(aperture)
            else:
                self._aperture = _zivid.Settings2D.Acquisition.Aperture()
            if brightness is not None:
                self._brightness = _zivid.Settings2D.Acquisition.Brightness(brightness)
            else:
                self._brightness = _zivid.Settings2D.Acquisition.Brightness()
            if exposure_time is not None:
                self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
                    exposure_time
                )
            else:
                self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime()
            if gain is not None:
                self._gain = _zivid.Settings2D.Acquisition.Gain(gain)
            else:
                self._gain = _zivid.Settings2D.Acquisition.Gain()

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
            self._aperture = _zivid.Settings2D.Acquisition.Aperture(value)

        @brightness.setter
        def brightness(self, value):
            self._brightness = _zivid.Settings2D.Acquisition.Brightness(value)

        @exposure_time.setter
        def exposure_time(self, value):
            self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(value)

        @gain.setter
        def gain(self, value):
            self._gain = _zivid.Settings2D.Acquisition.Gain(value)

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
                    blue=_zivid.Settings2D().Processing.Color.Balance().Blue().value,
                    green=_zivid.Settings2D().Processing.Color.Balance().Green().value,
                    red=_zivid.Settings2D().Processing.Color.Balance().Red().value,
                ):

                    if blue is not None:
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
                            blue
                        )
                    else:
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue()
                    if green is not None:
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(
                            green
                        )
                    else:
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green()
                    if red is not None:
                        self._red = _zivid.Settings2D.Processing.Color.Balance.Red(red)
                    else:
                        self._red = _zivid.Settings2D.Processing.Color.Balance.Red()

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
                    self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(value)

                @green.setter
                def green(self, value):
                    self._green = _zivid.Settings2D.Processing.Color.Balance.Green(
                        value
                    )

                @red.setter
                def red(self, value):
                    self._red = _zivid.Settings2D.Processing.Color.Balance.Red(value)

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
                    balance = zivid.Settings2D.Processing.Color.Balance()
                if not isinstance(balance, zivid.Settings2D.Processing.Color.Balance):
                    raise TypeError(
                        "Unsupported type: {value}".format(value=type(balance))
                    )
                self._balance = balance

            @property
            def balance(self):
                return self._balance

            @balance.setter
            def balance(self, value):
                if not isinstance(value, zivid.Settings2D.Processing.Color.Balance):
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

        def __init__(
            self, color=None,
        ):

            if color is None:
                color = zivid.Settings2D.Processing.Color()
            if not isinstance(color, zivid.Settings2D.Processing.Color):
                raise TypeError("Unsupported type: {value}".format(value=type(color)))
            self._color = color

        @property
        def color(self):
            return self._color

        @color.setter
        def color(self, value):
            if not isinstance(value, zivid.Settings2D.Processing.Color):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._color = value

        def __eq__(self, other):
            if self._color == other._color:
                return True
            return False

        def __str__(self):
            return """Processing:
        color: {color}
        """.format(
                color=self.color,
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
            processing = zivid.Settings2D.Processing()
        if not isinstance(processing, zivid.Settings2D.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

    @property
    def acquisitions(self):
        return self._acquisitions

    @property
    def processing(self):
        return self._processing

    @acquisitions.setter
    def acquisitions(self, value):
        from collections.abc import Iterable

        if not isinstance(value, Iterable):
            raise TypeError("Unsupported type: {value}".format(value=type(value)))
        self._acquisitions = _convert_to_acquistions(value)

    @processing.setter
    def processing(self, value):
        if not isinstance(value, zivid.Settings2D.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._processing == other._processing
        ):
            return True
        return False

    def __str__(self):
        return """Settings2D:
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
        if isinstance(acquisition_element, Settings2D.Acquisition):
            temp.append(acquisition_element)
        else:
            raise TypeError(
                "Unsupported type {type_of_acquisition_element}".format(
                    type_of_acquisition_element=type(acquisition_element)
                )
            )
    return temp


def _to_internal_acquisition(acquisition):
    internal_acquisition = _zivid.Settings2D.Acquisition()

    if acquisition.aperture is not None:

        internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture(
            acquisition.aperture
        )
    else:
        internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture()
    if acquisition.brightness is not None:

        internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness(
            acquisition.brightness
        )
    else:
        internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness()
    if acquisition.exposure_time is not None:

        internal_acquisition.exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
            acquisition.exposure_time
        )
    else:
        internal_acquisition.exposure_time = (
            _zivid.Settings2D.Acquisition.ExposureTime()
        )
    if acquisition.gain is not None:

        internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain(acquisition.gain)
    else:
        internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain()
    pass  # no children
    return internal_acquisition
