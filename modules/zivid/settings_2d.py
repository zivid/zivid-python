"""Auto generated, do not edit."""
# pylint: disable=missing-class-docstring,missing-function-docstring,line-too-long
import datetime
import collections.abc
import _zivid
import zivid
import zivid._settings2_d_converter


class Settings2D:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings2D().Acquisition().Aperture().value,
            brightness=_zivid.Settings2D().Acquisition().Brightness().value,
            exposure_time=_zivid.Settings2D().Acquisition().ExposureTime().value,
            gain=_zivid.Settings2D().Acquisition().Gain().value,
        ):

            if isinstance(aperture, (float, int,)) or aperture is None:
                self._aperture = _zivid.Settings2D.Acquisition.Aperture(aperture)
            else:
                raise TypeError(
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                        value_type=type(aperture)
                    )
                )
            if isinstance(brightness, (float, int,)) or brightness is None:
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
            if isinstance(gain, (float, int,)) or gain is None:
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
            if isinstance(value, (float, int,)) or value is None:
                self._aperture = _zivid.Settings2D.Acquisition.Aperture(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(
                        value_type=type(value)
                    )
                )

        @brightness.setter
        def brightness(self, value):
            if isinstance(value, (float, int,)) or value is None:
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
            if isinstance(value, (float, int,)) or value is None:
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
            return str(
                zivid._settings2_d_converter.to_internal_settings2_d_acquisition(self)
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

                    if isinstance(blue, (float, int,)) or blue is None:
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
                            blue
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(blue)
                            )
                        )
                    if isinstance(green, (float, int,)) or green is None:
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(
                            green
                        )
                    else:
                        raise TypeError(
                            "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                value_type=type(green)
                            )
                        )
                    if isinstance(red, (float, int,)) or red is None:
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
                    if isinstance(value, (float, int,)) or value is None:
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
                    if isinstance(value, (float, int,)) or value is None:
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
                    if isinstance(value, (float, int,)) or value is None:
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
                    return str(
                        zivid._settings2_d_converter.to_internal_settings2_d_processing_color_balance(
                            self
                        )
                    )

            def __init__(
                self,
                gamma=_zivid.Settings2D().Processing.Color().Gamma().value,
                balance=None,
            ):

                if isinstance(gamma, (float, int,)) or gamma is None:
                    self._gamma = _zivid.Settings2D.Processing.Color.Gamma(gamma)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                            value_type=type(gamma)
                        )
                    )
                if balance is None:
                    balance = zivid.Settings2D.Processing.Color.Balance()
                if not isinstance(balance, zivid.Settings2D.Processing.Color.Balance):
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
                    self._gamma = _zivid.Settings2D.Processing.Color.Gamma(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: float or  int or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @balance.setter
            def balance(self, value):
                if not isinstance(value, zivid.Settings2D.Processing.Color.Balance):
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
                    zivid._settings2_d_converter.to_internal_settings2_d_processing_color(
                        self
                    )
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
            return str(
                zivid._settings2_d_converter.to_internal_settings2_d_processing(self)
            )

    @property
    def processing(self):
        return self._processing

    @property
    def acquisitions(self):
        return self._acquisitions

    @acquisitions.setter
    def acquisitions(self, value):
        if not isinstance(value, collections.abc.Iterable):
            raise TypeError("Unsupported type: {value}".format(value=type(value)))
        self._acquisitions = _convert_to_acquistions(value)

    @processing.setter
    def processing(self, value):
        if not isinstance(value, zivid.Settings2D.Processing):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._processing = value

    def __str__(self):
        return str(zivid._settings2_d_converter.to_internal_settings2_d(self))

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
            processing = zivid.Settings2D.Processing()
        if not isinstance(processing, zivid.Settings2D.Processing):
            raise TypeError("Unsupported type: {value}".format(value=type(processing)))
        self._processing = processing

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._processing == other._processing
        ):
            return True
        return False


def _convert_to_acquistions(inputs):
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
