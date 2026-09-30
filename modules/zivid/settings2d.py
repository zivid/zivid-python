"""Auto generated, do not edit."""

from __future__ import annotations

import collections.abc

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime

import _zivid


class Settings2D:
    """Settings used when capturing 2D images with a Zivid camera."""

    class Acquisition:
        """Settings for one 2D acquisition. When capturing 2D HDR, all 2D acquisitions must have the same Aperture setting. Use Exposure Time or Gain to control the exposure instead."""

        def __init__(
            self,
            aperture: float | int | None = _zivid.Settings2D.Acquisition.Aperture().value,
            brightness: float | int | None = _zivid.Settings2D.Acquisition.Brightness().value,
            exposure_time: datetime.timedelta | None = _zivid.Settings2D.Acquisition.ExposureTime().value,
            gain: float | int | None = _zivid.Settings2D.Acquisition.Gain().value,
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

            if isinstance(exposure_time, (datetime.timedelta,)) or exposure_time is None:
                self._exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(exposure_time)
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
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(value_type=type(gain))
                )

        @property
        def aperture(self):
            """Aperture setting for the camera. Specified as an f-number (the ratio of lens focal length to the effective aperture diameter). When capturing 2D HDR, all 2D acquisitions must have the same Aperture setting. Use Exposure Time or Gain to control the exposure instead."""
            return self._aperture.value

        @property
        def brightness(self):
            """Brightness controls the light output from the projector. Brightness above 1.0 may be needed when the distance between the camera and the scene is large, or in case of high levels of ambient lighting. When brightness is above 1.0 the duty cycle of the camera (the percentage of time the camera can capture) will be reduced. The duty cycle in boost mode is 50%. The duty cycle is calculated over a 10 second period. This limitation is enforced automatically by the camera. Calling capture when the duty cycle limit has been reached will cause the camera to first wait (sleep) for a duration of time to cool down, before capture will start."""
            return self._brightness.value

        @property
        def exposure_time(self):
            """Exposure time for the image."""
            return self._exposure_time.value

        @property
        def gain(self):
            """Analog gain in the camera."""
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
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(value_type=type(value))
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
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(value_type=type(value))
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

    class Diagnostics:
        """When Diagnostics is enabled, additional diagnostic data is recorded during capture and included when saving the frame to a .zdf file. This enables Zivid's Customer Success team to provide better assistance and more thorough troubleshooting. Enabling Diagnostics increases the capture time and the RAM usage. It will also increase the size of the .zdf file. It is recommended to enable Diagnostics only when reporting issues to Zivid's support team."""

        def __init__(
            self,
            enabled: bool | None = _zivid.Settings2D.Diagnostics.Enabled().value,
        ):

            if isinstance(enabled, (bool,)) or enabled is None:
                self._enabled = _zivid.Settings2D.Diagnostics.Enabled(enabled)
            else:
                raise TypeError(
                    "Unsupported type, expected: (bool,) or None, got {value_type}".format(value_type=type(enabled))
                )

        @property
        def enabled(self):
            """Enable or disable diagnostics."""
            return self._enabled.value

        @enabled.setter
        def enabled(self, value):
            if isinstance(value, (bool,)) or value is None:
                self._enabled = _zivid.Settings2D.Diagnostics.Enabled(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: bool or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._enabled == other._enabled:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings2d_diagnostics(self))

    class Processing:
        """2D processing settings."""

        class Color:
            """Color settings."""

            class Balance:
                """Color balance settings."""

                def __init__(
                    self,
                    blue: float | int | None = _zivid.Settings2D.Processing.Color.Balance.Blue().value,
                    green: float | int | None = _zivid.Settings2D.Processing.Color.Balance.Green().value,
                    red: float | int | None = _zivid.Settings2D.Processing.Color.Balance.Red().value,
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
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(blue)
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
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(green)
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
                    """Digital gain applied to blue channel."""
                    return self._blue.value

                @property
                def green(self):
                    """Digital gain applied to green channel."""
                    return self._green.value

                @property
                def red(self):
                    """Digital gain applied to red channel."""
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
                        self._blue = _zivid.Settings2D.Processing.Color.Balance.Blue(value)
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
                        self._green = _zivid.Settings2D.Processing.Color.Balance.Green(value)
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
                        self._red = _zivid.Settings2D.Processing.Color.Balance.Red(value)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                value_type=type(value)
                            )
                        )

                def __eq__(self, other):
                    if self._blue == other._blue and self._green == other._green and self._red == other._red:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings2d_processing_color_balance(self))

            class Experimental:
                """Experimental color settings. These may be renamed, moved or deleted in the future."""

                class Mode:
                    """This setting controls how the color image is computed. `automatic` is the default option. It performs tone mapping for HDR captures, but not for single-acquisition captures. Use this mode with a single acquisition if you want to have the most control over the colors in the image. `toneMapping` uses all the acquisitions to create one merged and normalized color image. For HDR captures the dynamic range of the captured images is usually higher than the 8-bit color image range. `toneMapping` will map the HDR color data to the 8-bit color output range by applying a scaling factor. `toneMapping` can also be used for single-acquisition captures to normalize the captured color image to the full 8-bit output. Note that when using `toneMapping` mode the color values can be inconsistent over repeated captures if you move, add or remove objects in the scene. For the most control over the colors in the single-acquisition case, select the `automatic` mode."""

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
                    mode: str | None = _zivid.Settings2D.Processing.Color.Experimental.Mode().value,
                ):

                    if isinstance(mode, _zivid.Settings2D.Processing.Color.Experimental.Mode.enum) or mode is None:
                        self._mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(mode)
                    elif isinstance(mode, str):
                        self._mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(self.Mode._valid_values[mode])
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(mode))
                        )

                @property
                def mode(self):
                    """This setting controls how the color image is computed. `automatic` is the default option. It performs tone mapping for HDR captures, but not for single-acquisition captures. Use this mode with a single acquisition if you want to have the most control over the colors in the image. `toneMapping` uses all the acquisitions to create one merged and normalized color image. For HDR captures the dynamic range of the captured images is usually higher than the 8-bit color image range. `toneMapping` will map the HDR color data to the 8-bit color output range by applying a scaling factor. `toneMapping` can also be used for single-acquisition captures to normalize the captured color image to the full 8-bit output. Note that when using `toneMapping` mode the color values can be inconsistent over repeated captures if you move, add or remove objects in the scene. For the most control over the colors in the single-acquisition case, select the `automatic` mode."""
                    if self._mode.value is None:
                        return None
                    for key, internal_value in self.Mode._valid_values.items():
                        if internal_value == self._mode.value:
                            return key
                    raise ValueError("Unsupported value {value}".format(value=self._mode))

                @mode.setter
                def mode(self, value):
                    if isinstance(value, str):
                        self._mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(
                            self.Mode._valid_values[value]
                        )
                    elif isinstance(value, _zivid.Settings2D.Processing.Color.Experimental.Mode.enum) or value is None:
                        self._mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(value)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                        )

                def __eq__(self, other):
                    if self._mode == other._mode:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings2d_processing_color_experimental(self))

            def __init__(
                self,
                gamma: float | int | None = _zivid.Settings2D.Processing.Color.Gamma().value,
                balance: Settings2D.Processing.Color.Balance | None = None,
                experimental: Settings2D.Processing.Color.Experimental | None = None,
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
                    raise TypeError("Unsupported type: {value}".format(value=type(balance)))
                self._balance = balance

                if experimental is None:
                    experimental = self.Experimental()
                if not isinstance(experimental, self.Experimental):
                    raise TypeError("Unsupported type: {value}".format(value=type(experimental)))
                self._experimental = experimental

            @property
            def gamma(self):
                """Gamma applied to the color values. Gamma less than 1 makes the colors brighter, while gamma greater than 1 makes the colors darker."""
                return self._gamma.value

            @property
            def balance(self):
                """Color balance settings."""
                return self._balance

            @property
            def experimental(self):
                """Experimental color settings. These may be renamed, moved or deleted in the future."""
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
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._balance = value

            @experimental.setter
            def experimental(self, value):
                if not isinstance(value, self.Experimental):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
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
            color: Settings2D.Processing.Color | None = None,
        ):

            if color is None:
                color = self.Color()
            if not isinstance(color, self.Color):
                raise TypeError("Unsupported type: {value}".format(value=type(color)))
            self._color = color

        @property
        def color(self):
            """Color settings."""
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
        """Sampling settings."""

        class Interval:
            """Sampling interval controls the interval between successive sensor operations (e.g., structured light pattern projection and image exposure), aligned to external frequencies (e.g., 50 Hz, 60 Hz grid) or to other devices (e.g., barcode scanners at 100 Hz). The requested interval is a target: the sensor operations will happen at this rate if the it can fit the chosen exposure time plus some processing overhead. Otherwise, the sampling interval is rounded up to the nearest suitable integer multiple (e.g., n * 10 ms for 100 Hz and n * 8.33 ms for 120 Hz)."""

            def __init__(
                self,
                duration: datetime.timedelta | None = _zivid.Settings2D.Sampling.Interval.Duration().value,
                enabled: bool | None = _zivid.Settings2D.Sampling.Interval.Enabled().value,
            ):

                if isinstance(duration, (datetime.timedelta,)) or duration is None:
                    self._duration = _zivid.Settings2D.Sampling.Interval.Duration(duration)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (datetime.timedelta,) or None, got {value_type}".format(
                            value_type=type(duration)
                        )
                    )

                if isinstance(enabled, (bool,)) or enabled is None:
                    self._enabled = _zivid.Settings2D.Sampling.Interval.Enabled(enabled)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (bool,) or None, got {value_type}".format(value_type=type(enabled))
                    )

            @property
            def duration(self):
                """Duration between successive sensor operations, in microseconds. The effective interval might be rounded up to the nearest suitable integer multiple and will never be shorter than exposure time plus some processing overhead."""
                return self._duration.value

            @property
            def enabled(self):
                """Enable or disable sampling interval."""
                return self._enabled.value

            @duration.setter
            def duration(self, value):
                if isinstance(value, (datetime.timedelta,)) or value is None:
                    self._duration = _zivid.Settings2D.Sampling.Interval.Duration(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: datetime.timedelta or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @enabled.setter
            def enabled(self, value):
                if isinstance(value, (bool,)) or value is None:
                    self._enabled = _zivid.Settings2D.Sampling.Interval.Enabled(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: bool or None, got {value_type}".format(value_type=type(value))
                    )

            def __eq__(self, other):
                if self._duration == other._duration and self._enabled == other._enabled:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings2d_sampling_interval(self))

        class Color:
            """Choose how to sample colors for the 2D image. - `rgb` option gives an image with full colors. - `grayscale` option gives a grayscale (r=g=b) image, which can be acquired faster than full colors. - `rgbStrongAmbientLight` option gives an image with full colors and reduced color noise. This option should be chosen only for applications which suffer from high color noise and with high amounts of ambient light in the scene. - `rgbAmbientSuppression` option gives an image with full colors while suppressing the ambient light. The Zivid 2+R and Zivid 3 cameras suppress ambient light by default, and therefore do not need the additional option `rgbAmbientSuppression`. The `grayscale`, `rgbStrongAmbientLight` and `rgbAmbientSuppression` options are not available on all camera models."""

            grayscale = "grayscale"
            rgb = "rgb"
            rgbAmbientSuppression = "rgbAmbientSuppression"
            rgbStrongAmbientLight = "rgbStrongAmbientLight"

            _valid_values = {
                "grayscale": _zivid.Settings2D.Sampling.Color.grayscale,
                "rgb": _zivid.Settings2D.Sampling.Color.rgb,
                "rgbAmbientSuppression": _zivid.Settings2D.Sampling.Color.rgbAmbientSuppression,
                "rgbStrongAmbientLight": _zivid.Settings2D.Sampling.Color.rgbStrongAmbientLight,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        class Pixel:
            """Set the pixel sampling to use for the 2D capture. This setting defines how the camera sensor is sampled. When doing 2D+3D capture, picking the same value that is used for 3D is generally recommended."""

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
            color: str | None = _zivid.Settings2D.Sampling.Color().value,
            pixel: str | None = _zivid.Settings2D.Sampling.Pixel().value,
            interval: Settings2D.Sampling.Interval | None = None,
        ):

            if isinstance(color, _zivid.Settings2D.Sampling.Color.enum) or color is None:
                self._color = _zivid.Settings2D.Sampling.Color(color)
            elif isinstance(color, str):
                self._color = _zivid.Settings2D.Sampling.Color(self.Color._valid_values[color])
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(color))
                )

            if isinstance(pixel, _zivid.Settings2D.Sampling.Pixel.enum) or pixel is None:
                self._pixel = _zivid.Settings2D.Sampling.Pixel(pixel)
            elif isinstance(pixel, str):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(self.Pixel._valid_values[pixel])
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(pixel))
                )

            if interval is None:
                interval = self.Interval()
            if not isinstance(interval, self.Interval):
                raise TypeError("Unsupported type: {value}".format(value=type(interval)))
            self._interval = interval

        @property
        def color(self):
            """Choose how to sample colors for the 2D image. - `rgb` option gives an image with full colors. - `grayscale` option gives a grayscale (r=g=b) image, which can be acquired faster than full colors. - `rgbStrongAmbientLight` option gives an image with full colors and reduced color noise. This option should be chosen only for applications which suffer from high color noise and with high amounts of ambient light in the scene. - `rgbAmbientSuppression` option gives an image with full colors while suppressing the ambient light. The Zivid 2+R and Zivid 3 cameras suppress ambient light by default, and therefore do not need the additional option `rgbAmbientSuppression`. The `grayscale`, `rgbStrongAmbientLight` and `rgbAmbientSuppression` options are not available on all camera models."""
            if self._color.value is None:
                return None
            for key, internal_value in self.Color._valid_values.items():
                if internal_value == self._color.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._color))

        @property
        def pixel(self):
            """Set the pixel sampling to use for the 2D capture. This setting defines how the camera sensor is sampled. When doing 2D+3D capture, picking the same value that is used for 3D is generally recommended."""
            if self._pixel.value is None:
                return None
            for key, internal_value in self.Pixel._valid_values.items():
                if internal_value == self._pixel.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._pixel))

        @property
        def interval(self):
            """Sampling interval controls the interval between successive sensor operations (e.g., structured light pattern projection and image exposure), aligned to external frequencies (e.g., 50 Hz, 60 Hz grid) or to other devices (e.g., barcode scanners at 100 Hz). The requested interval is a target: the sensor operations will happen at this rate if the it can fit the chosen exposure time plus some processing overhead. Otherwise, the sampling interval is rounded up to the nearest suitable integer multiple (e.g., n * 10 ms for 100 Hz and n * 8.33 ms for 120 Hz)."""
            return self._interval

        @color.setter
        def color(self, value):
            if isinstance(value, str):
                self._color = _zivid.Settings2D.Sampling.Color(self.Color._valid_values[value])
            elif isinstance(value, _zivid.Settings2D.Sampling.Color.enum) or value is None:
                self._color = _zivid.Settings2D.Sampling.Color(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        @pixel.setter
        def pixel(self, value):
            if isinstance(value, str):
                self._pixel = _zivid.Settings2D.Sampling.Pixel(self.Pixel._valid_values[value])
            elif isinstance(value, _zivid.Settings2D.Sampling.Pixel.enum) or value is None:
                self._pixel = _zivid.Settings2D.Sampling.Pixel(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        @interval.setter
        def interval(self, value):
            if not isinstance(value, self.Interval):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._interval = value

        def __eq__(self, other):
            if self._color == other._color and self._pixel == other._pixel and self._interval == other._interval:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings2d_sampling(self))

    def __init__(
        self,
        acquisitions=None,
        diagnostics: Settings2D.Diagnostics | None = None,
        processing: Settings2D.Processing | None = None,
        sampling: Settings2D.Sampling | None = None,
    ):

        if acquisitions is None:
            self._acquisitions = []
        elif isinstance(acquisitions, (collections.abc.Iterable,)):
            self._acquisitions = []
            for item in acquisitions:
                if isinstance(item, self.Acquisition):
                    self._acquisitions.append(item)
                else:
                    raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))
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
        """List of acquisitions used for 2D capture."""
        return self._acquisitions

    @property
    def diagnostics(self):
        """When Diagnostics is enabled, additional diagnostic data is recorded during capture and included when saving the frame to a .zdf file. This enables Zivid's Customer Success team to provide better assistance and more thorough troubleshooting. Enabling Diagnostics increases the capture time and the RAM usage. It will also increase the size of the .zdf file. It is recommended to enable Diagnostics only when reporting issues to Zivid's support team."""
        return self._diagnostics

    @property
    def processing(self):
        """2D processing settings."""
        return self._processing

    @property
    def sampling(self):
        """Sampling settings."""
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
                raise TypeError("Unsupported type {item_type}".format(item_type=type(item)))

    @diagnostics.setter
    def diagnostics(self, value):
        if not isinstance(value, self.Diagnostics):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._diagnostics = value

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
            and self._diagnostics == other._diagnostics
            and self._processing == other._processing
            and self._sampling == other._sampling
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_settings2d(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_settings2d(_to_internal_settings2d(self))


def _to_settings2d_acquisition(internal_acquisition):
    return Settings2D.Acquisition(
        aperture=internal_acquisition.aperture.value,
        brightness=internal_acquisition.brightness.value,
        exposure_time=internal_acquisition.exposure_time.value,
        gain=internal_acquisition.gain.value,
    )


def _to_settings2d_diagnostics(internal_diagnostics):
    return Settings2D.Diagnostics(
        enabled=internal_diagnostics.enabled.value,
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
        experimental=_to_settings2d_processing_color_experimental(internal_color.experimental),
        gamma=internal_color.gamma.value,
    )


def _to_settings2d_processing(internal_processing):
    return Settings2D.Processing(
        color=_to_settings2d_processing_color(internal_processing.color),
    )


def _to_settings2d_sampling_interval(internal_interval):
    return Settings2D.Sampling.Interval(
        duration=internal_interval.duration.value,
        enabled=internal_interval.enabled.value,
    )


def _to_settings2d_sampling(internal_sampling):
    return Settings2D.Sampling(
        interval=_to_settings2d_sampling_interval(internal_sampling.interval),
        color=internal_sampling.color.value,
        pixel=internal_sampling.pixel.value,
    )


def _to_settings2d(internal_settings2d):
    return Settings2D(
        acquisitions=[_to_settings2d_acquisition(value) for value in internal_settings2d.acquisitions.value],
        diagnostics=_to_settings2d_diagnostics(internal_settings2d.diagnostics),
        processing=_to_settings2d_processing(internal_settings2d.processing),
        sampling=_to_settings2d_sampling(internal_settings2d.sampling),
    )


def _to_internal_settings2d_acquisition(acquisition):
    internal_acquisition = _zivid.Settings2D.Acquisition()

    internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture(acquisition.aperture)
    internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness(acquisition.brightness)
    internal_acquisition.exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(acquisition.exposure_time)
    internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain(acquisition.gain)

    return internal_acquisition


def _to_internal_settings2d_diagnostics(diagnostics):
    internal_diagnostics = _zivid.Settings2D.Diagnostics()

    internal_diagnostics.enabled = _zivid.Settings2D.Diagnostics.Enabled(diagnostics.enabled)

    return internal_diagnostics


def _to_internal_settings2d_processing_color_balance(balance):
    internal_balance = _zivid.Settings2D.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings2D.Processing.Color.Balance.Blue(balance.blue)
    internal_balance.green = _zivid.Settings2D.Processing.Color.Balance.Green(balance.green)
    internal_balance.red = _zivid.Settings2D.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def _to_internal_settings2d_processing_color_experimental(experimental):
    internal_experimental = _zivid.Settings2D.Processing.Color.Experimental()

    internal_experimental.mode = _zivid.Settings2D.Processing.Color.Experimental.Mode(experimental._mode.value)

    return internal_experimental


def _to_internal_settings2d_processing_color(color):
    internal_color = _zivid.Settings2D.Processing.Color()

    internal_color.gamma = _zivid.Settings2D.Processing.Color.Gamma(color.gamma)

    internal_color.balance = _to_internal_settings2d_processing_color_balance(color.balance)
    internal_color.experimental = _to_internal_settings2d_processing_color_experimental(color.experimental)
    return internal_color


def _to_internal_settings2d_processing(processing):
    internal_processing = _zivid.Settings2D.Processing()

    internal_processing.color = _to_internal_settings2d_processing_color(processing.color)
    return internal_processing


def _to_internal_settings2d_sampling_interval(interval):
    internal_interval = _zivid.Settings2D.Sampling.Interval()

    internal_interval.duration = _zivid.Settings2D.Sampling.Interval.Duration(interval.duration)
    internal_interval.enabled = _zivid.Settings2D.Sampling.Interval.Enabled(interval.enabled)

    return internal_interval


def _to_internal_settings2d_sampling(sampling):
    internal_sampling = _zivid.Settings2D.Sampling()

    internal_sampling.color = _zivid.Settings2D.Sampling.Color(sampling._color.value)
    internal_sampling.pixel = _zivid.Settings2D.Sampling.Pixel(sampling._pixel.value)

    internal_sampling.interval = _to_internal_settings2d_sampling_interval(sampling.interval)
    return internal_sampling


def _to_internal_settings2d(settings2d):
    internal_settings2d = _zivid.Settings2D()

    temp_acquisitions = _zivid.Settings2D.Acquisitions()
    for value in settings2d.acquisitions:
        temp_acquisitions.append(_to_internal_settings2d_acquisition(value))
    internal_settings2d.acquisitions = temp_acquisitions

    internal_settings2d.diagnostics = _to_internal_settings2d_diagnostics(settings2d.diagnostics)
    internal_settings2d.processing = _to_internal_settings2d_processing(settings2d.processing)
    internal_settings2d.sampling = _to_internal_settings2d_sampling(settings2d.sampling)
    return internal_settings2d
