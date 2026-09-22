"""Auto generated, do not edit."""

from __future__ import annotations

import collections.abc

# pylint: disable=too-many-lines,protected-access,too-few-public-methods,too-many-arguments,too-many-positional-arguments,line-too-long,missing-function-docstring,missing-class-docstring,redefined-builtin,too-many-branches,too-many-boolean-expressions
import datetime

import _zivid
import zivid.settings2d


class Settings:
    """Settings used when capturing a 3D capture or 2D+3D capture with a Zivid camera."""

    class Acquisition:
        """Settings for a single acquisition."""

        def __init__(
            self,
            aperture: float | int | None = _zivid.Settings.Acquisition.Aperture().value,
            brightness: float | int | None = _zivid.Settings.Acquisition.Brightness().value,
            exposure_time: datetime.timedelta | None = _zivid.Settings.Acquisition.ExposureTime().value,
            gain: float | int | None = _zivid.Settings.Acquisition.Gain().value,
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

            if isinstance(exposure_time, (datetime.timedelta,)) or exposure_time is None:
                self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(exposure_time)
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
                    "Unsupported type, expected: (float, int,) or None, got {value_type}".format(value_type=type(gain))
                )

        @property
        def aperture(self):
            """Aperture setting for the camera. Specified as an f-number (the ratio of lens focal length to the effective aperture diameter)."""
            return self._aperture.value

        @property
        def brightness(self):
            """Brightness controls the light output from the projector. Brightness above 1.0 may be needed when the distance between the camera and the scene is large, or in case of high levels of ambient lighting. When brightness is above 1.0 the duty cycle of the camera (the percentage of time the camera can capture) will be reduced. The duty cycle in boost mode is 50%. The duty cycle is calculated over a 10 second period. This limitation is enforced automatically by the camera. Calling capture when the duty cycle limit has been reached will cause the camera to first wait (sleep) for a duration of time to cool down, before capture will start."""
            return self._brightness.value

        @property
        def exposure_time(self):
            """Exposure time for each single image in the measurement. Affects frame rate."""
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
                self._aperture = _zivid.Settings.Acquisition.Aperture(value)
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
                self._brightness = _zivid.Settings.Acquisition.Brightness(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: float or  int or None, got {value_type}".format(value_type=type(value))
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
            return str(_to_internal_settings_acquisition(self))

    class Diagnostics:
        """When Diagnostics is enabled, additional diagnostic data is recorded during capture and included when saving the frame to a .zdf file. This enables Zivid's Customer Success team to provide better assistance and more thorough troubleshooting. Enabling Diagnostics increases the capture time and the RAM usage. It will also increase the size of the .zdf file. It is recommended to enable Diagnostics only when reporting issues to Zivid's support team."""

        def __init__(
            self,
            enabled: bool | None = _zivid.Settings.Diagnostics.Enabled().value,
        ):

            if isinstance(enabled, (bool,)) or enabled is None:
                self._enabled = _zivid.Settings.Diagnostics.Enabled(enabled)
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
                self._enabled = _zivid.Settings.Diagnostics.Enabled(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: bool or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._enabled == other._enabled:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_diagnostics(self))

    class Processing:
        """Settings related to processing of a capture, including filters and color balance."""

        class Color:
            """Color settings. These settings are deprecated as of SDK 2.14. These settings will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""

            class Balance:
                """Color balance settings."""

                def __init__(
                    self,
                    blue: float | int | None = _zivid.Settings.Processing.Color.Balance.Blue().value,
                    green: float | int | None = _zivid.Settings.Processing.Color.Balance.Green().value,
                    red: float | int | None = _zivid.Settings.Processing.Color.Balance.Red().value,
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
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(green)
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
                    """Digital gain applied to blue channel. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
                    return self._blue.value

                @property
                def green(self):
                    """Digital gain applied to green channel. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
                    return self._green.value

                @property
                def red(self):
                    """Digital gain applied to red channel. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
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
                        self._blue = _zivid.Settings.Processing.Color.Balance.Blue(value)
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
                        self._green = _zivid.Settings.Processing.Color.Balance.Green(value)
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
                    if self._blue == other._blue and self._green == other._green and self._red == other._red:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_color_balance(self))

            class Experimental:
                """Experimental color settings. These may be renamed, moved or deleted in the future."""

                class Mode:
                    """This setting controls how the color image is computed. `automatic` is the default option. `automatic` is identical to `useFirstAcquisition` for single-acquisition captures and multi-acquisition captures when all the acquisitions have identical (duplicated) acquisition settings. `automatic` is identical to `toneMapping` for multi-acquisition HDR captures with differing acquisition settings. `useFirstAcquisition` uses the color data acquired from the first acquisition provided. If the capture consists of more than one acquisition, then the remaining acquisitions are not used for the color image. No tone mapping is performed. This option provides the most control of the color image, and the color values will be consistent over repeated captures with the same settings. `toneMapping` uses all the acquisitions to create one merged and normalized color image. For HDR captures the dynamic range of the captured images is usually higher than the 8-bit color image range. `toneMapping` will map the HDR color data to the 8-bit color output range by applying a scaling factor. `toneMapping` can also be used for single-acquisition captures to normalize the captured color image to the full 8-bit output. Note that when using `toneMapping` mode the color values can be inconsistent over repeated captures if you move, add or remove objects in the scene. For the most control over the colors, select the `useFirstAcquisition` mode. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""

                    automatic = "automatic"
                    toneMapping = "toneMapping"
                    useFirstAcquisition = "useFirstAcquisition"

                    _valid_values = {
                        "automatic": _zivid.Settings.Processing.Color.Experimental.Mode.automatic,
                        "toneMapping": _zivid.Settings.Processing.Color.Experimental.Mode.toneMapping,
                        "useFirstAcquisition": _zivid.Settings.Processing.Color.Experimental.Mode.useFirstAcquisition,
                    }

                    @classmethod
                    def valid_values(cls):
                        return list(cls._valid_values.keys())

                def __init__(
                    self,
                    mode: str | None = _zivid.Settings.Processing.Color.Experimental.Mode().value,
                ):

                    if isinstance(mode, _zivid.Settings.Processing.Color.Experimental.Mode.enum) or mode is None:
                        self._mode = _zivid.Settings.Processing.Color.Experimental.Mode(mode)
                    elif isinstance(mode, str):
                        self._mode = _zivid.Settings.Processing.Color.Experimental.Mode(self.Mode._valid_values[mode])
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(mode))
                        )

                @property
                def mode(self):
                    """This setting controls how the color image is computed. `automatic` is the default option. `automatic` is identical to `useFirstAcquisition` for single-acquisition captures and multi-acquisition captures when all the acquisitions have identical (duplicated) acquisition settings. `automatic` is identical to `toneMapping` for multi-acquisition HDR captures with differing acquisition settings. `useFirstAcquisition` uses the color data acquired from the first acquisition provided. If the capture consists of more than one acquisition, then the remaining acquisitions are not used for the color image. No tone mapping is performed. This option provides the most control of the color image, and the color values will be consistent over repeated captures with the same settings. `toneMapping` uses all the acquisitions to create one merged and normalized color image. For HDR captures the dynamic range of the captured images is usually higher than the 8-bit color image range. `toneMapping` will map the HDR color data to the 8-bit color output range by applying a scaling factor. `toneMapping` can also be used for single-acquisition captures to normalize the captured color image to the full 8-bit output. Note that when using `toneMapping` mode the color values can be inconsistent over repeated captures if you move, add or remove objects in the scene. For the most control over the colors, select the `useFirstAcquisition` mode. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
                    if self._mode.value is None:
                        return None
                    for key, internal_value in self.Mode._valid_values.items():
                        if internal_value == self._mode.value:
                            return key
                    raise ValueError("Unsupported value {value}".format(value=self._mode))

                @mode.setter
                def mode(self, value):
                    if isinstance(value, str):
                        self._mode = _zivid.Settings.Processing.Color.Experimental.Mode(self.Mode._valid_values[value])
                    elif isinstance(value, _zivid.Settings.Processing.Color.Experimental.Mode.enum) or value is None:
                        self._mode = _zivid.Settings.Processing.Color.Experimental.Mode(value)
                    else:
                        raise TypeError(
                            "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                        )

                def __eq__(self, other):
                    if self._mode == other._mode:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_color_experimental(self))

            def __init__(
                self,
                gamma: float | int | None = _zivid.Settings.Processing.Color.Gamma().value,
                balance: Settings.Processing.Color.Balance | None = None,
                experimental: Settings.Processing.Color.Experimental | None = None,
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
                    raise TypeError("Unsupported type: {value}".format(value=type(balance)))
                self._balance = balance

                if experimental is None:
                    experimental = self.Experimental()
                if not isinstance(experimental, self.Experimental):
                    raise TypeError("Unsupported type: {value}".format(value=type(experimental)))
                self._experimental = experimental

            @property
            def gamma(self):
                """Gamma applied to the color values. Gamma less than 1 makes the colors brighter, while gamma greater than 1 makes the colors darker. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
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
                return str(_to_internal_settings_processing_color(self))

        class Filters:
            """Filter settings."""

            class Cluster:
                """Removes floating points and isolated clusters from the point cloud."""

                class Removal:
                    """Cluster removal filter."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Cluster.Removal.Enabled().value,
                        max_neighbor_distance: (
                            float | int | None
                        ) = _zivid.Settings.Processing.Filters.Cluster.Removal.MaxNeighborDistance().value,
                        min_area: (
                            float | int | None
                        ) = _zivid.Settings.Processing.Filters.Cluster.Removal.MinArea().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Cluster.Removal.Enabled(enabled)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                        if (
                            isinstance(
                                max_neighbor_distance,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or max_neighbor_distance is None
                        ):
                            self._max_neighbor_distance = (
                                _zivid.Settings.Processing.Filters.Cluster.Removal.MaxNeighborDistance(
                                    max_neighbor_distance
                                )
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(max_neighbor_distance)
                                )
                            )

                        if (
                            isinstance(
                                min_area,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or min_area is None
                        ):
                            self._min_area = _zivid.Settings.Processing.Filters.Cluster.Removal.MinArea(min_area)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(min_area)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable cluster removal."""
                        return self._enabled.value

                    @property
                    def max_neighbor_distance(self):
                        """Maximum normalized distance between neighboring points that are still classified as belonging to the same cluster. The default value is optimal for most scenes. On messy scenes turning this setting down helps removing more bad points."""
                        return self._max_neighbor_distance.value

                    @property
                    def min_area(self):
                        """Clusters with area below this threshold are removed by the filter. The area is given in mm^2."""
                        return self._min_area.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Cluster.Removal.Enabled(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @max_neighbor_distance.setter
                    def max_neighbor_distance(self, value):
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
                            self._max_neighbor_distance = (
                                _zivid.Settings.Processing.Filters.Cluster.Removal.MaxNeighborDistance(value)
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @min_area.setter
                    def min_area(self, value):
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
                            self._min_area = _zivid.Settings.Processing.Filters.Cluster.Removal.MinArea(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._max_neighbor_distance == other._max_neighbor_distance
                            and self._min_area == other._min_area
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_cluster_removal(self))

                def __init__(
                    self,
                    removal: Settings.Processing.Filters.Cluster.Removal | None = None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError("Unsupported type: {value}".format(value=type(removal)))
                    self._removal = removal

                @property
                def removal(self):
                    """Cluster removal filter."""
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_cluster(self))

            class Experimental:
                """Experimental filters. These may be renamed, moved or deleted in the future."""

                class ContrastDistortion:
                    """Corrects artifacts that appear when imaging scenes with large texture gradients or high contrast. These artifacts are caused by blurring in the lens. The filter works best when aperture values are chosen such that the camera has quite good focus. The filter also supports removing the points that experience a large correction."""

                    class Correction:
                        """Contrast distortion correction filter."""

                        def __init__(
                            self,
                            enabled: (
                                bool | None
                            ) = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled().value,
                            strength: (
                                float | int | None
                            ) = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength().value,
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
                            """Enable or disable contrast distortion correction."""
                            return self._enabled.value

                        @property
                        def strength(self):
                            """Strength of correction. Higher values give more correction."""
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
                            if self._enabled == other._enabled and self._strength == other._strength:
                                return True
                            return False

                        def __str__(self):
                            return str(
                                _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
                                    self
                                )
                            )

                    class Removal:
                        """Contrast distortion removal filter."""

                        def __init__(
                            self,
                            enabled: (
                                bool | None
                            ) = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled().value,
                            threshold: (
                                float | int | None
                            ) = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold().value,
                        ):

                            if isinstance(enabled, (bool,)) or enabled is None:
                                self._enabled = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                        enabled
                                    )
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
                            """Enable or disable contrast distortion removal."""
                            return self._enabled.value

                        @property
                        def threshold(self):
                            """Threshold for removal. Higher values remove more points."""
                            return self._threshold.value

                        @enabled.setter
                        def enabled(self, value):
                            if isinstance(value, (bool,)) or value is None:
                                self._enabled = (
                                    _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
                                        value
                                    )
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
                            if self._enabled == other._enabled and self._threshold == other._threshold:
                                return True
                            return False

                        def __str__(self):
                            return str(
                                _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(self)
                            )

                    def __init__(
                        self,
                        correction: (
                            Settings.Processing.Filters.Experimental.ContrastDistortion.Correction | None
                        ) = None,
                        removal: Settings.Processing.Filters.Experimental.ContrastDistortion.Removal | None = None,
                    ):

                        if correction is None:
                            correction = self.Correction()
                        if not isinstance(correction, self.Correction):
                            raise TypeError("Unsupported type: {value}".format(value=type(correction)))
                        self._correction = correction

                        if removal is None:
                            removal = self.Removal()
                        if not isinstance(removal, self.Removal):
                            raise TypeError("Unsupported type: {value}".format(value=type(removal)))
                        self._removal = removal

                    @property
                    def correction(self):
                        """Contrast distortion correction filter."""
                        return self._correction

                    @property
                    def removal(self):
                        """Contrast distortion removal filter."""
                        return self._removal

                    @correction.setter
                    def correction(self, value):
                        if not isinstance(value, self.Correction):
                            raise TypeError("Unsupported type {value}".format(value=type(value)))
                        self._correction = value

                    @removal.setter
                    def removal(self, value):
                        if not isinstance(value, self.Removal):
                            raise TypeError("Unsupported type {value}".format(value=type(value)))
                        self._removal = value

                    def __eq__(self, other):
                        if self._correction == other._correction and self._removal == other._removal:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_experimental_contrast_distortion(self))

                def __init__(
                    self,
                    contrast_distortion: Settings.Processing.Filters.Experimental.ContrastDistortion | None = None,
                ):

                    if contrast_distortion is None:
                        contrast_distortion = self.ContrastDistortion()
                    if not isinstance(contrast_distortion, self.ContrastDistortion):
                        raise TypeError("Unsupported type: {value}".format(value=type(contrast_distortion)))
                    self._contrast_distortion = contrast_distortion

                @property
                def contrast_distortion(self):
                    """Corrects artifacts that appear when imaging scenes with large texture gradients or high contrast. These artifacts are caused by blurring in the lens. The filter works best when aperture values are chosen such that the camera has quite good focus. The filter also supports removing the points that experience a large correction."""
                    return self._contrast_distortion

                @contrast_distortion.setter
                def contrast_distortion(self, value):
                    if not isinstance(value, self.ContrastDistortion):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._contrast_distortion = value

                def __eq__(self, other):
                    if self._contrast_distortion == other._contrast_distortion:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_experimental(self))

            class Hole:
                """Contains filters that can be used to deal with holes in the point cloud."""

                class Repair:
                    """Fills in point cloud holes by interpolating remaining surrounding points."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Hole.Repair.Enabled().value,
                        hole_size: float | int | None = _zivid.Settings.Processing.Filters.Hole.Repair.HoleSize().value,
                        strictness: int | None = _zivid.Settings.Processing.Filters.Hole.Repair.Strictness().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Hole.Repair.Enabled(enabled)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                        if (
                            isinstance(
                                hole_size,
                                (
                                    float,
                                    int,
                                ),
                            )
                            or hole_size is None
                        ):
                            self._hole_size = _zivid.Settings.Processing.Filters.Hole.Repair.HoleSize(hole_size)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(hole_size)
                                )
                            )

                        if isinstance(strictness, (int,)) or strictness is None:
                            self._strictness = _zivid.Settings.Processing.Filters.Hole.Repair.Strictness(strictness)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (int,) or None, got {value_type}".format(
                                    value_type=type(strictness)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable hole repair."""
                        return self._enabled.value

                    @property
                    def hole_size(self):
                        """Relative diameter of holes to fill. Increasing this will fill more points, but require more computation time. The maximum allowed hole size scales with distance, so that we allow filling larger holes at greater distances, measured in mm."""
                        return self._hole_size.value

                    @property
                    def strictness(self):
                        """Level of strictness when considering if a point should be filled. A higher level of strictness requires a missing point to be surrounded by valid points on more sides in order to be filled. Increasing this will fill fewer points, but it will be less likely to fill gaps that are not circular, for example between two edges."""
                        return self._strictness.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Hole.Repair.Enabled(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @hole_size.setter
                    def hole_size(self, value):
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
                            self._hole_size = _zivid.Settings.Processing.Filters.Hole.Repair.HoleSize(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @strictness.setter
                    def strictness(self, value):
                        if isinstance(value, (int,)) or value is None:
                            self._strictness = _zivid.Settings.Processing.Filters.Hole.Repair.Strictness(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if (
                            self._enabled == other._enabled
                            and self._hole_size == other._hole_size
                            and self._strictness == other._strictness
                        ):
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_hole_repair(self))

                def __init__(
                    self,
                    repair: Settings.Processing.Filters.Hole.Repair | None = None,
                ):

                    if repair is None:
                        repair = self.Repair()
                    if not isinstance(repair, self.Repair):
                        raise TypeError("Unsupported type: {value}".format(value=type(repair)))
                    self._repair = repair

                @property
                def repair(self):
                    """Fills in point cloud holes by interpolating remaining surrounding points."""
                    return self._repair

                @repair.setter
                def repair(self, value):
                    if not isinstance(value, self.Repair):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._repair = value

                def __eq__(self, other):
                    if self._repair == other._repair:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_hole(self))

            class Noise:
                """Contains filters that can be used to clean up a noisy point cloud."""

                class Removal:
                    """Discard points with signal-to-noise ratio (SNR) values below a threshold."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled().value,
                        threshold: (
                            float | int | None
                        ) = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(enabled)
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
                            self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(threshold)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(threshold)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable the SNR filter."""
                        return self._enabled.value

                    @property
                    def threshold(self):
                        """Discard points with signal-to-noise ratio (SNR) below the given value."""
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(value)
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
                            self._threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled and self._threshold == other._threshold:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_noise_removal(self))

                class Repair:
                    """Get better surface coverage by repairing regions of missing data due to noisy points. Consider disabling this filter if you require all points in your point cloud to be of high confidence."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Noise.Repair.Enabled().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Repair.Enabled(enabled)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable noise repair."""
                        return self._enabled.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Repair.Enabled(value)
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
                        return str(_to_internal_settings_processing_filters_noise_repair(self))

                class Suppression:
                    """Reduce noise and outliers in the point cloud. This filter can also be used to reduce ripple effects caused by interreflections. Consider disabling this filter if you need to distinguish very fine details and thus need to avoid any smoothing effects."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Noise.Suppression.Enabled().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Suppression.Enabled(enabled)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable noise suppression."""
                        return self._enabled.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Noise.Suppression.Enabled(value)
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
                        return str(_to_internal_settings_processing_filters_noise_suppression(self))

                def __init__(
                    self,
                    removal: Settings.Processing.Filters.Noise.Removal | None = None,
                    repair: Settings.Processing.Filters.Noise.Repair | None = None,
                    suppression: Settings.Processing.Filters.Noise.Suppression | None = None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError("Unsupported type: {value}".format(value=type(removal)))
                    self._removal = removal

                    if repair is None:
                        repair = self.Repair()
                    if not isinstance(repair, self.Repair):
                        raise TypeError("Unsupported type: {value}".format(value=type(repair)))
                    self._repair = repair

                    if suppression is None:
                        suppression = self.Suppression()
                    if not isinstance(suppression, self.Suppression):
                        raise TypeError("Unsupported type: {value}".format(value=type(suppression)))
                    self._suppression = suppression

                @property
                def removal(self):
                    """Discard points with signal-to-noise ratio (SNR) values below a threshold."""
                    return self._removal

                @property
                def repair(self):
                    """Get better surface coverage by repairing regions of missing data due to noisy points. Consider disabling this filter if you require all points in your point cloud to be of high confidence."""
                    return self._repair

                @property
                def suppression(self):
                    """Reduce noise and outliers in the point cloud. This filter can also be used to reduce ripple effects caused by interreflections. Consider disabling this filter if you need to distinguish very fine details and thus need to avoid any smoothing effects."""
                    return self._suppression

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._removal = value

                @repair.setter
                def repair(self, value):
                    if not isinstance(value, self.Repair):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._repair = value

                @suppression.setter
                def suppression(self, value):
                    if not isinstance(value, self.Suppression):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._suppression = value

                def __eq__(self, other):
                    if (
                        self._removal == other._removal
                        and self._repair == other._repair
                        and self._suppression == other._suppression
                    ):
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_noise(self))

            class Outlier:
                """Contains a filter that removes points with large Euclidean distance to neighboring points."""

                class Removal:
                    """Discard point if Euclidean distance to neighboring points is above a threshold."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled().value,
                        threshold: (
                            float | int | None
                        ) = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(enabled)
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
                            self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(threshold)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(threshold)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable the outlier filter."""
                        return self._enabled.value

                    @property
                    def threshold(self):
                        """Discard point if Euclidean distance to neighboring points is above the given value."""
                        return self._threshold.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(value)
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
                            self._threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled and self._threshold == other._threshold:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_outlier_removal(self))

                def __init__(
                    self,
                    removal: Settings.Processing.Filters.Outlier.Removal | None = None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError("Unsupported type: {value}".format(value=type(removal)))
                    self._removal = removal

                @property
                def removal(self):
                    """Discard point if Euclidean distance to neighboring points is above a threshold."""
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_outlier(self))

            class Reflection:
                """Contains a filter that removes points likely introduced by reflections (useful for shiny materials)."""

                class Removal:
                    """Discard points likely introduced by reflections (useful for shiny materials)."""

                    class Mode:
                        """The reflection filter has two modes: Local and Global. Global mode is generally better at removing outlier points in the point cloud, whereas Local mode preserves more 3D data on thin objects far from the background. Additionally, Local mode behaves slightly differently depending on the Engine used. With Phase engine or Stripe engine, Local mode will process faster than Global mode and remove more reflection artifacts. Also with Stripe engine, Local mode might reduce the number of points resolved on shiny and reflective objects. With Omni engine, Local mode will reduce the amount of artifacts in low projector signal regions, but might also reduce coverage on surfaces with low projector signal. It is advised to use the Cluster filter together with Local mode."""

                        global_ = "global"
                        local = "local"

                        _valid_values = {
                            "global": _zivid.Settings.Processing.Filters.Reflection.Removal.Mode.global_,
                            "local": _zivid.Settings.Processing.Filters.Reflection.Removal.Mode.local,
                        }

                        @classmethod
                        def valid_values(cls):
                            return list(cls._valid_values.keys())

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled().value,
                        mode: str | None = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(enabled)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (bool,) or None, got {value_type}".format(
                                    value_type=type(enabled)
                                )
                            )

                        if (
                            isinstance(mode, _zivid.Settings.Processing.Filters.Reflection.Removal.Mode.enum)
                            or mode is None
                        ):
                            self._mode = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode(mode)
                        elif isinstance(mode, str):
                            self._mode = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode(
                                self.Mode._valid_values[mode]
                            )
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str or None, got {value_type}".format(
                                    value_type=type(mode)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable the reflection filter. Note that this filter is computationally intensive and may affect the frame rate."""
                        return self._enabled.value

                    @property
                    def mode(self):
                        """The reflection filter has two modes: Local and Global. Global mode is generally better at removing outlier points in the point cloud, whereas Local mode preserves more 3D data on thin objects far from the background. Additionally, Local mode behaves slightly differently depending on the Engine used. With Phase engine or Stripe engine, Local mode will process faster than Global mode and remove more reflection artifacts. Also with Stripe engine, Local mode might reduce the number of points resolved on shiny and reflective objects. With Omni engine, Local mode will reduce the amount of artifacts in low projector signal regions, but might also reduce coverage on surfaces with low projector signal. It is advised to use the Cluster filter together with Local mode."""
                        if self._mode.value is None:
                            return None
                        for key, internal_value in self.Mode._valid_values.items():
                            if internal_value == self._mode.value:
                                return key
                        raise ValueError("Unsupported value {value}".format(value=self._mode))

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: bool or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    @mode.setter
                    def mode(self, value):
                        if isinstance(value, str):
                            self._mode = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode(
                                self.Mode._valid_values[value]
                            )
                        elif (
                            isinstance(value, _zivid.Settings.Processing.Filters.Reflection.Removal.Mode.enum)
                            or value is None
                        ):
                            self._mode = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: str or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled and self._mode == other._mode:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_reflection_removal(self))

                def __init__(
                    self,
                    removal: Settings.Processing.Filters.Reflection.Removal | None = None,
                ):

                    if removal is None:
                        removal = self.Removal()
                    if not isinstance(removal, self.Removal):
                        raise TypeError("Unsupported type: {value}".format(value=type(removal)))
                    self._removal = removal

                @property
                def removal(self):
                    """Discard points likely introduced by reflections (useful for shiny materials)."""
                    return self._removal

                @removal.setter
                def removal(self, value):
                    if not isinstance(value, self.Removal):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._removal = value

                def __eq__(self, other):
                    if self._removal == other._removal:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_reflection(self))

            class Smoothing:
                """Smoothing filters."""

                class Gaussian:
                    """Gaussian smoothing of the point cloud."""

                    def __init__(
                        self,
                        enabled: bool | None = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled().value,
                        sigma: float | int | None = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma().value,
                    ):

                        if isinstance(enabled, (bool,)) or enabled is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(enabled)
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
                            self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(sigma)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: (float, int,) or None, got {value_type}".format(
                                    value_type=type(sigma)
                                )
                            )

                    @property
                    def enabled(self):
                        """Enable or disable the smoothing filter."""
                        return self._enabled.value

                    @property
                    def sigma(self):
                        """Higher values result in smoother point clouds (Standard deviation of the filter coefficients)."""
                        return self._sigma.value

                    @enabled.setter
                    def enabled(self, value):
                        if isinstance(value, (bool,)) or value is None:
                            self._enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(value)
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
                            self._sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(value)
                        else:
                            raise TypeError(
                                "Unsupported type, expected: float or  int or None, got {value_type}".format(
                                    value_type=type(value)
                                )
                            )

                    def __eq__(self, other):
                        if self._enabled == other._enabled and self._sigma == other._sigma:
                            return True
                        return False

                    def __str__(self):
                        return str(_to_internal_settings_processing_filters_smoothing_gaussian(self))

                def __init__(
                    self,
                    gaussian: Settings.Processing.Filters.Smoothing.Gaussian | None = None,
                ):

                    if gaussian is None:
                        gaussian = self.Gaussian()
                    if not isinstance(gaussian, self.Gaussian):
                        raise TypeError("Unsupported type: {value}".format(value=type(gaussian)))
                    self._gaussian = gaussian

                @property
                def gaussian(self):
                    """Gaussian smoothing of the point cloud."""
                    return self._gaussian

                @gaussian.setter
                def gaussian(self, value):
                    if not isinstance(value, self.Gaussian):
                        raise TypeError("Unsupported type {value}".format(value=type(value)))
                    self._gaussian = value

                def __eq__(self, other):
                    if self._gaussian == other._gaussian:
                        return True
                    return False

                def __str__(self):
                    return str(_to_internal_settings_processing_filters_smoothing(self))

            def __init__(
                self,
                cluster: Settings.Processing.Filters.Cluster | None = None,
                experimental: Settings.Processing.Filters.Experimental | None = None,
                hole: Settings.Processing.Filters.Hole | None = None,
                noise: Settings.Processing.Filters.Noise | None = None,
                outlier: Settings.Processing.Filters.Outlier | None = None,
                reflection: Settings.Processing.Filters.Reflection | None = None,
                smoothing: Settings.Processing.Filters.Smoothing | None = None,
            ):

                if cluster is None:
                    cluster = self.Cluster()
                if not isinstance(cluster, self.Cluster):
                    raise TypeError("Unsupported type: {value}".format(value=type(cluster)))
                self._cluster = cluster

                if experimental is None:
                    experimental = self.Experimental()
                if not isinstance(experimental, self.Experimental):
                    raise TypeError("Unsupported type: {value}".format(value=type(experimental)))
                self._experimental = experimental

                if hole is None:
                    hole = self.Hole()
                if not isinstance(hole, self.Hole):
                    raise TypeError("Unsupported type: {value}".format(value=type(hole)))
                self._hole = hole

                if noise is None:
                    noise = self.Noise()
                if not isinstance(noise, self.Noise):
                    raise TypeError("Unsupported type: {value}".format(value=type(noise)))
                self._noise = noise

                if outlier is None:
                    outlier = self.Outlier()
                if not isinstance(outlier, self.Outlier):
                    raise TypeError("Unsupported type: {value}".format(value=type(outlier)))
                self._outlier = outlier

                if reflection is None:
                    reflection = self.Reflection()
                if not isinstance(reflection, self.Reflection):
                    raise TypeError("Unsupported type: {value}".format(value=type(reflection)))
                self._reflection = reflection

                if smoothing is None:
                    smoothing = self.Smoothing()
                if not isinstance(smoothing, self.Smoothing):
                    raise TypeError("Unsupported type: {value}".format(value=type(smoothing)))
                self._smoothing = smoothing

            @property
            def cluster(self):
                """Removes floating points and isolated clusters from the point cloud."""
                return self._cluster

            @property
            def experimental(self):
                """Experimental filters. These may be renamed, moved or deleted in the future."""
                return self._experimental

            @property
            def hole(self):
                """Contains filters that can be used to deal with holes in the point cloud."""
                return self._hole

            @property
            def noise(self):
                """Contains filters that can be used to clean up a noisy point cloud."""
                return self._noise

            @property
            def outlier(self):
                """Contains a filter that removes points with large Euclidean distance to neighboring points."""
                return self._outlier

            @property
            def reflection(self):
                """Contains a filter that removes points likely introduced by reflections (useful for shiny materials)."""
                return self._reflection

            @property
            def smoothing(self):
                """Smoothing filters."""
                return self._smoothing

            @cluster.setter
            def cluster(self, value):
                if not isinstance(value, self.Cluster):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._cluster = value

            @experimental.setter
            def experimental(self, value):
                if not isinstance(value, self.Experimental):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._experimental = value

            @hole.setter
            def hole(self, value):
                if not isinstance(value, self.Hole):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._hole = value

            @noise.setter
            def noise(self, value):
                if not isinstance(value, self.Noise):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._noise = value

            @outlier.setter
            def outlier(self, value):
                if not isinstance(value, self.Outlier):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._outlier = value

            @reflection.setter
            def reflection(self, value):
                if not isinstance(value, self.Reflection):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._reflection = value

            @smoothing.setter
            def smoothing(self, value):
                if not isinstance(value, self.Smoothing):
                    raise TypeError("Unsupported type {value}".format(value=type(value)))
                self._smoothing = value

            def __eq__(self, other):
                if (
                    self._cluster == other._cluster
                    and self._experimental == other._experimental
                    and self._hole == other._hole
                    and self._noise == other._noise
                    and self._outlier == other._outlier
                    and self._reflection == other._reflection
                    and self._smoothing == other._smoothing
                ):
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings_processing_filters(self))

        class Resampling:
            """Settings for changing the output resolution of the point cloud."""

            class Mode:
                """Setting for upsampling or downsampling the point cloud data by some factor. This operation is performed after all other processing has been completed. Downsampling is used to reduce the number of points in the point cloud. This is done by combining each 2x2 or 4x4 group of pixels in the original point cloud into one pixel in a new point cloud. This downsample functionality is identical to the downsample method on the PointCloud class. The averaging process reduces noise in the point cloud, but it will not improve capture speed. To improve capture speed, consider using the subsampling modes found in Settings/Sampling/Pixel. Upsampling is used to increase the number of points in the point cloud. It is not possible to upsample beyond the full resolution of the camera, so upsampling may only be used in combination with the subsampling modes found in Settings/Sampling/Pixel. For example, one may combine blueSubsample2x2 with upsample2x2 to obtain a point cloud that matches a full resolution 2D capture, while retaining the speed benefits of capturing the point cloud with blueSubsample2x2. Upsampling is achieved by expanding pixels in the original point cloud into groups of 2x2 or 4x4 pixels in a new point cloud. Where possible, values are filled at the new points based on an interpolation of the surrounding original points. The points in the new point cloud that correspond to points in the original point cloud are left unchanged. Note that upsampling will lead to four (upsample2x2) or sixteen (upsample4x4) times as many pixels in the point cloud compared to no upsampling, so users should be aware of increased computational cost related to copying and analyzing this data."""

                disabled = "disabled"
                downsample2x2 = "downsample2x2"
                downsample4x4 = "downsample4x4"
                upsample2x2 = "upsample2x2"
                upsample4x4 = "upsample4x4"

                _valid_values = {
                    "disabled": _zivid.Settings.Processing.Resampling.Mode.disabled,
                    "downsample2x2": _zivid.Settings.Processing.Resampling.Mode.downsample2x2,
                    "downsample4x4": _zivid.Settings.Processing.Resampling.Mode.downsample4x4,
                    "upsample2x2": _zivid.Settings.Processing.Resampling.Mode.upsample2x2,
                    "upsample4x4": _zivid.Settings.Processing.Resampling.Mode.upsample4x4,
                }

                @classmethod
                def valid_values(cls):
                    return list(cls._valid_values.keys())

            def __init__(
                self,
                mode: str | None = _zivid.Settings.Processing.Resampling.Mode().value,
            ):

                if isinstance(mode, _zivid.Settings.Processing.Resampling.Mode.enum) or mode is None:
                    self._mode = _zivid.Settings.Processing.Resampling.Mode(mode)
                elif isinstance(mode, str):
                    self._mode = _zivid.Settings.Processing.Resampling.Mode(self.Mode._valid_values[mode])
                else:
                    raise TypeError(
                        "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(mode))
                    )

            @property
            def mode(self):
                """Setting for upsampling or downsampling the point cloud data by some factor. This operation is performed after all other processing has been completed. Downsampling is used to reduce the number of points in the point cloud. This is done by combining each 2x2 or 4x4 group of pixels in the original point cloud into one pixel in a new point cloud. This downsample functionality is identical to the downsample method on the PointCloud class. The averaging process reduces noise in the point cloud, but it will not improve capture speed. To improve capture speed, consider using the subsampling modes found in Settings/Sampling/Pixel. Upsampling is used to increase the number of points in the point cloud. It is not possible to upsample beyond the full resolution of the camera, so upsampling may only be used in combination with the subsampling modes found in Settings/Sampling/Pixel. For example, one may combine blueSubsample2x2 with upsample2x2 to obtain a point cloud that matches a full resolution 2D capture, while retaining the speed benefits of capturing the point cloud with blueSubsample2x2. Upsampling is achieved by expanding pixels in the original point cloud into groups of 2x2 or 4x4 pixels in a new point cloud. Where possible, values are filled at the new points based on an interpolation of the surrounding original points. The points in the new point cloud that correspond to points in the original point cloud are left unchanged. Note that upsampling will lead to four (upsample2x2) or sixteen (upsample4x4) times as many pixels in the point cloud compared to no upsampling, so users should be aware of increased computational cost related to copying and analyzing this data."""
                if self._mode.value is None:
                    return None
                for key, internal_value in self.Mode._valid_values.items():
                    if internal_value == self._mode.value:
                        return key
                raise ValueError("Unsupported value {value}".format(value=self._mode))

            @mode.setter
            def mode(self, value):
                if isinstance(value, str):
                    self._mode = _zivid.Settings.Processing.Resampling.Mode(self.Mode._valid_values[value])
                elif isinstance(value, _zivid.Settings.Processing.Resampling.Mode.enum) or value is None:
                    self._mode = _zivid.Settings.Processing.Resampling.Mode(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                    )

            def __eq__(self, other):
                if self._mode == other._mode:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings_processing_resampling(self))

        def __init__(
            self,
            color: Settings.Processing.Color | None = None,
            filters: Settings.Processing.Filters | None = None,
            resampling: Settings.Processing.Resampling | None = None,
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

            if resampling is None:
                resampling = self.Resampling()
            if not isinstance(resampling, self.Resampling):
                raise TypeError("Unsupported type: {value}".format(value=type(resampling)))
            self._resampling = resampling

        @property
        def color(self):
            """Color settings. These settings are deprecated as of SDK 2.14. These settings will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` node. Tip: If you want to convert an existing settings.yml file to use the `Settings/Color` node, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
            return self._color

        @property
        def filters(self):
            """Filter settings."""
            return self._filters

        @property
        def resampling(self):
            """Settings for changing the output resolution of the point cloud."""
            return self._resampling

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

        @resampling.setter
        def resampling(self, value):
            if not isinstance(value, self.Resampling):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._resampling = value

        def __eq__(self, other):
            if (
                self._color == other._color
                and self._filters == other._filters
                and self._resampling == other._resampling
            ):
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_processing(self))

    class RegionOfInterest:
        """Removes points outside the region of interest."""

        class Box:
            """Removes points outside the given three-dimensional box. Using this feature may significantly speed up acquisition and processing time, because one can avoid acquiring and processing data that is guaranteed to fall outside of the region of interest. The degree of speed-up depends on the size and shape of the box. Generally, a smaller box yields a greater speed-up. The box is defined by three points: O, A and B. These points define two vectors, OA that goes from PointO to PointA, and OB that goes from PointO to PointB. This gives 4 points O, A, B and (O + OA + OB), that together form a parallelogram in 3D. Two extents can be provided, to extrude the parallelogram along the surface normal vector of the parallelogram plane. This creates a 3D volume (parallelepiped). The surface normal vector is defined by the cross product OA x OB."""

            def __init__(
                self,
                enabled: bool | None = _zivid.Settings.RegionOfInterest.Box.Enabled().value,
                extents=_zivid.Settings.RegionOfInterest.Box.Extents().value,
                point_a=_zivid.Settings.RegionOfInterest.Box.PointA().value,
                point_b=_zivid.Settings.RegionOfInterest.Box.PointB().value,
                point_o=_zivid.Settings.RegionOfInterest.Box.PointO().value,
            ):

                if isinstance(enabled, (bool,)) or enabled is None:
                    self._enabled = _zivid.Settings.RegionOfInterest.Box.Enabled(enabled)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (bool,) or None, got {value_type}".format(value_type=type(enabled))
                    )

                if isinstance(extents, (collections.abc.Iterable, _zivid.data_model.Range)) or extents is None:
                    self._extents = _zivid.Settings.RegionOfInterest.Box.Extents(extents)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (collections.abc.Iterable, _zivid.data_model.Range) or None, got {value_type}".format(
                            value_type=type(extents)
                        )
                    )

                if isinstance(point_a, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or point_a is None:
                    self._point_a = _zivid.Settings.RegionOfInterest.Box.PointA(point_a)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (collections.abc.Iterable, _zivid.data_model.PointXYZ) or None, got {value_type}".format(
                            value_type=type(point_a)
                        )
                    )

                if isinstance(point_b, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or point_b is None:
                    self._point_b = _zivid.Settings.RegionOfInterest.Box.PointB(point_b)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (collections.abc.Iterable, _zivid.data_model.PointXYZ) or None, got {value_type}".format(
                            value_type=type(point_b)
                        )
                    )

                if isinstance(point_o, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or point_o is None:
                    self._point_o = _zivid.Settings.RegionOfInterest.Box.PointO(point_o)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (collections.abc.Iterable, _zivid.data_model.PointXYZ) or None, got {value_type}".format(
                            value_type=type(point_o)
                        )
                    )

            @property
            def enabled(self):
                """Enable or disable box filter."""
                return self._enabled.value

            @property
            def extents(self):
                """Two points on the normal describing the direction and distance from the plane from which the normal is derived."""
                if self._extents.value is None:
                    return None
                return self._extents.value.to_array()

            @property
            def point_a(self):
                """A point such that the vector from PointO to PointA describes the first edge of the parallelogram."""
                if self._point_a.value is None:
                    return None
                return self._point_a.value.to_array()

            @property
            def point_b(self):
                """A point such that the vector from PointO to PointB describes the second edge of the parallelogram."""
                if self._point_b.value is None:
                    return None
                return self._point_b.value.to_array()

            @property
            def point_o(self):
                """The point at the intersection of two adjacent edges defining a parallelogram."""
                if self._point_o.value is None:
                    return None
                return self._point_o.value.to_array()

            @enabled.setter
            def enabled(self, value):
                if isinstance(value, (bool,)) or value is None:
                    self._enabled = _zivid.Settings.RegionOfInterest.Box.Enabled(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: bool or None, got {value_type}".format(value_type=type(value))
                    )

            @extents.setter
            def extents(self, value):
                if isinstance(value, (collections.abc.Iterable, _zivid.data_model.Range)) or value is None:
                    self._extents = _zivid.Settings.RegionOfInterest.Box.Extents(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: collections.abc.Iterable or  _zivid.data_model.Rang or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @point_a.setter
            def point_a(self, value):
                if isinstance(value, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or value is None:
                    self._point_a = _zivid.Settings.RegionOfInterest.Box.PointA(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: collections.abc.Iterable or  _zivid.data_model.PointXY or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @point_b.setter
            def point_b(self, value):
                if isinstance(value, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or value is None:
                    self._point_b = _zivid.Settings.RegionOfInterest.Box.PointB(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: collections.abc.Iterable or  _zivid.data_model.PointXY or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            @point_o.setter
            def point_o(self, value):
                if isinstance(value, (collections.abc.Iterable, _zivid.data_model.PointXYZ)) or value is None:
                    self._point_o = _zivid.Settings.RegionOfInterest.Box.PointO(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: collections.abc.Iterable or  _zivid.data_model.PointXY or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if (
                    self._enabled == other._enabled
                    and self._extents == other._extents
                    and self._point_a == other._point_a
                    and self._point_b == other._point_b
                    and self._point_o == other._point_o
                ):
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings_region_of_interest_box(self))

        class Depth:
            """Removes points that reside outside of a depth range, meaning that their Z coordinate falls above a given maximum or below a given minimum."""

            def __init__(
                self,
                enabled: bool | None = _zivid.Settings.RegionOfInterest.Depth.Enabled().value,
                range=_zivid.Settings.RegionOfInterest.Depth.Range().value,
            ):

                if isinstance(enabled, (bool,)) or enabled is None:
                    self._enabled = _zivid.Settings.RegionOfInterest.Depth.Enabled(enabled)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (bool,) or None, got {value_type}".format(value_type=type(enabled))
                    )

                if isinstance(range, (collections.abc.Iterable, _zivid.data_model.Range)) or range is None:
                    self._range = _zivid.Settings.RegionOfInterest.Depth.Range(range)
                else:
                    raise TypeError(
                        "Unsupported type, expected: (collections.abc.Iterable, _zivid.data_model.Range) or None, got {value_type}".format(
                            value_type=type(range)
                        )
                    )

            @property
            def enabled(self):
                """Enable or disable depth filter."""
                return self._enabled.value

            @property
            def range(self):
                """Specify the minimum and maximum Z value that will be included."""
                if self._range.value is None:
                    return None
                return self._range.value.to_array()

            @enabled.setter
            def enabled(self, value):
                if isinstance(value, (bool,)) or value is None:
                    self._enabled = _zivid.Settings.RegionOfInterest.Depth.Enabled(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: bool or None, got {value_type}".format(value_type=type(value))
                    )

            @range.setter
            def range(self, value):
                if isinstance(value, (collections.abc.Iterable, _zivid.data_model.Range)) or value is None:
                    self._range = _zivid.Settings.RegionOfInterest.Depth.Range(value)
                else:
                    raise TypeError(
                        "Unsupported type, expected: collections.abc.Iterable or  _zivid.data_model.Rang or None, got {value_type}".format(
                            value_type=type(value)
                        )
                    )

            def __eq__(self, other):
                if self._enabled == other._enabled and self._range == other._range:
                    return True
                return False

            def __str__(self):
                return str(_to_internal_settings_region_of_interest_depth(self))

        def __init__(
            self,
            box: Settings.RegionOfInterest.Box | None = None,
            depth: Settings.RegionOfInterest.Depth | None = None,
        ):

            if box is None:
                box = self.Box()
            if not isinstance(box, self.Box):
                raise TypeError("Unsupported type: {value}".format(value=type(box)))
            self._box = box

            if depth is None:
                depth = self.Depth()
            if not isinstance(depth, self.Depth):
                raise TypeError("Unsupported type: {value}".format(value=type(depth)))
            self._depth = depth

        @property
        def box(self):
            """Removes points outside the given three-dimensional box. Using this feature may significantly speed up acquisition and processing time, because one can avoid acquiring and processing data that is guaranteed to fall outside of the region of interest. The degree of speed-up depends on the size and shape of the box. Generally, a smaller box yields a greater speed-up. The box is defined by three points: O, A and B. These points define two vectors, OA that goes from PointO to PointA, and OB that goes from PointO to PointB. This gives 4 points O, A, B and (O + OA + OB), that together form a parallelogram in 3D. Two extents can be provided, to extrude the parallelogram along the surface normal vector of the parallelogram plane. This creates a 3D volume (parallelepiped). The surface normal vector is defined by the cross product OA x OB."""
            return self._box

        @property
        def depth(self):
            """Removes points that reside outside of a depth range, meaning that their Z coordinate falls above a given maximum or below a given minimum."""
            return self._depth

        @box.setter
        def box(self, value):
            if not isinstance(value, self.Box):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._box = value

        @depth.setter
        def depth(self, value):
            if not isinstance(value, self.Depth):
                raise TypeError("Unsupported type {value}".format(value=type(value)))
            self._depth = value

        def __eq__(self, other):
            if self._box == other._box and self._depth == other._depth:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_region_of_interest(self))

    class Sampling:
        """Sampling settings."""

        class Color:
            """Choose how to sample colors for the point cloud. - `rgb` option gives an image with full colors. - `grayscale` option gives a grayscale (r=g=b) image, which can be acquired faster than full colors. - `rgbStrongAmbientLight` option gives an image with full colors and reduced color noise. This option should be chosen only for applications which suffer from high color noise and with high amounts of ambient light in the scene. - `rgbAmbientSuppression` option gives an image with full colors while suppressing the ambient light. The Zivid 2+R and Zivid 3 cameras suppress ambient light by default, and therefore do not need the additional option `rgbAmbientSuppression`. The `grayscale`, `rgbStrongAmbientLight` and `rgbAmbientSuppression` options are not available on all camera models. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` field. Tip: If you want to convert an existing settings.yml file to the recommended API, you can import the .yml file into Zivid Studio and then re-export it to .yml."""

            disabled = "disabled"
            grayscale = "grayscale"
            rgb = "rgb"
            rgbAmbientSuppression = "rgbAmbientSuppression"
            rgbStrongAmbientLight = "rgbStrongAmbientLight"

            _valid_values = {
                "disabled": _zivid.Settings.Sampling.Color.disabled,
                "grayscale": _zivid.Settings.Sampling.Color.grayscale,
                "rgb": _zivid.Settings.Sampling.Color.rgb,
                "rgbAmbientSuppression": _zivid.Settings.Sampling.Color.rgbAmbientSuppression,
                "rgbStrongAmbientLight": _zivid.Settings.Sampling.Color.rgbStrongAmbientLight,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        class Pixel:
            """For Zivid 2 and Zivid 2+, this setting controls whether to read out the full image sensor and use white projector light or to subsample pixels for specific color channels with corresponding projector light. Picking a specific color channel can help reduce noise and effects of ambient light - projecting blue light is recommended. For Zivid 2+R and Zivid 3, the user does not have to set the projection color and should only consider whether to scale the resolution `by2x2` or `by4x4`. Both of these modes will boost the signal strength by about 4x compared to `all`, so the user should consider a corresponding reduction in exposure time. Sampling at a decreased resolution decreases capture time, as less data will be captured and processed."""

            all = "all"
            blueSubsample2x2 = "blueSubsample2x2"
            blueSubsample4x4 = "blueSubsample4x4"
            by2x2 = "by2x2"
            by4x4 = "by4x4"
            redSubsample2x2 = "redSubsample2x2"
            redSubsample4x4 = "redSubsample4x4"

            _valid_values = {
                "all": _zivid.Settings.Sampling.Pixel.all,
                "blueSubsample2x2": _zivid.Settings.Sampling.Pixel.blueSubsample2x2,
                "blueSubsample4x4": _zivid.Settings.Sampling.Pixel.blueSubsample4x4,
                "by2x2": _zivid.Settings.Sampling.Pixel.by2x2,
                "by4x4": _zivid.Settings.Sampling.Pixel.by4x4,
                "redSubsample2x2": _zivid.Settings.Sampling.Pixel.redSubsample2x2,
                "redSubsample4x4": _zivid.Settings.Sampling.Pixel.redSubsample4x4,
            }

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())

        def __init__(
            self,
            color: str | None = _zivid.Settings.Sampling.Color().value,
            pixel: str | None = _zivid.Settings.Sampling.Pixel().value,
        ):

            if isinstance(color, _zivid.Settings.Sampling.Color.enum) or color is None:
                self._color = _zivid.Settings.Sampling.Color(color)
            elif isinstance(color, str):
                self._color = _zivid.Settings.Sampling.Color(self.Color._valid_values[color])
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(color))
                )

            if isinstance(pixel, _zivid.Settings.Sampling.Pixel.enum) or pixel is None:
                self._pixel = _zivid.Settings.Sampling.Pixel(pixel)
            elif isinstance(pixel, str):
                self._pixel = _zivid.Settings.Sampling.Pixel(self.Pixel._valid_values[pixel])
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(pixel))
                )

        @property
        def color(self):
            """Choose how to sample colors for the point cloud. - `rgb` option gives an image with full colors. - `grayscale` option gives a grayscale (r=g=b) image, which can be acquired faster than full colors. - `rgbStrongAmbientLight` option gives an image with full colors and reduced color noise. This option should be chosen only for applications which suffer from high color noise and with high amounts of ambient light in the scene. - `rgbAmbientSuppression` option gives an image with full colors while suppressing the ambient light. The Zivid 2+R and Zivid 3 cameras suppress ambient light by default, and therefore do not need the additional option `rgbAmbientSuppression`. The `grayscale`, `rgbStrongAmbientLight` and `rgbAmbientSuppression` options are not available on all camera models. This setting is deprecated as of SDK 2.14. This setting will be removed in the next SDK major version (SDK 3.0). The recommended way to do a 2D+3D capture is to set a `Settings2D` object in the `Settings/Color` field. Tip: If you want to convert an existing settings.yml file to the recommended API, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
            if self._color.value is None:
                return None
            for key, internal_value in self.Color._valid_values.items():
                if internal_value == self._color.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._color))

        @property
        def pixel(self):
            """For Zivid 2 and Zivid 2+, this setting controls whether to read out the full image sensor and use white projector light or to subsample pixels for specific color channels with corresponding projector light. Picking a specific color channel can help reduce noise and effects of ambient light - projecting blue light is recommended. For Zivid 2+R and Zivid 3, the user does not have to set the projection color and should only consider whether to scale the resolution `by2x2` or `by4x4`. Both of these modes will boost the signal strength by about 4x compared to `all`, so the user should consider a corresponding reduction in exposure time. Sampling at a decreased resolution decreases capture time, as less data will be captured and processed."""
            if self._pixel.value is None:
                return None
            for key, internal_value in self.Pixel._valid_values.items():
                if internal_value == self._pixel.value:
                    return key
            raise ValueError("Unsupported value {value}".format(value=self._pixel))

        @color.setter
        def color(self, value):
            if isinstance(value, str):
                self._color = _zivid.Settings.Sampling.Color(self.Color._valid_values[value])
            elif isinstance(value, _zivid.Settings.Sampling.Color.enum) or value is None:
                self._color = _zivid.Settings.Sampling.Color(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        @pixel.setter
        def pixel(self, value):
            if isinstance(value, str):
                self._pixel = _zivid.Settings.Sampling.Pixel(self.Pixel._valid_values[value])
            elif isinstance(value, _zivid.Settings.Sampling.Pixel.enum) or value is None:
                self._pixel = _zivid.Settings.Sampling.Pixel(value)
            else:
                raise TypeError(
                    "Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value))
                )

        def __eq__(self, other):
            if self._color == other._color and self._pixel == other._pixel:
                return True
            return False

        def __str__(self):
            return str(_to_internal_settings_sampling(self))

    class Engine:
        """Set the Zivid Vision Engine to use. The Phase Engine is the fastest choice in terms of both acquisition time and total capture time, and is a good compromise between quality and speed. The Phase Engine is recommended for objects that are diffuse, opaque, and slightly specular, and is suitable for applications in logistics such as parcel induction. The Stripe Engine is built for exceptional point cloud quality in scenes with highly specular reflective objects. This makes the engine suitable for applications such as factory automation, manufacturing, and bin picking. Additional acquisition and processing time are required for the Stripe Engine. The Omni Engine is built for exceptional point cloud quality on all scenes, including scenes with extremely specular reflective objects, as well as transparent objects. This makes the Omni Engine suitable for applications such as piece picking. Same as for the Stripe Engine, it trades off speed for quality. The Omni Engine is only available for Zivid 2+ and Zivid 2+R. The Sage engine is built for use cases that require all points to be correct/accurate with particularly high confidence. This can be very suitable for eliminating problems such as reflection artifacts. This validation comes at the cost of speed, and sometimes reduced number of valid points due to the removal of low-confidence data. The Sage Engine is only available for Zivid 2+R."""

        omni = "omni"
        phase = "phase"
        sage = "sage"
        stripe = "stripe"

        _valid_values = {
            "omni": _zivid.Settings.Engine.omni,
            "phase": _zivid.Settings.Engine.phase,
            "sage": _zivid.Settings.Engine.sage,
            "stripe": _zivid.Settings.Engine.stripe,
        }

        @classmethod
        def valid_values(cls):
            return list(cls._valid_values.keys())

    def __init__(
        self,
        acquisitions=None,
        color: zivid.settings2d.Settings2D | None = None,
        engine: str | None = _zivid.Settings.Engine().value,
        diagnostics: Settings.Diagnostics | None = None,
        processing: Settings.Processing | None = None,
        region_of_interest: Settings.RegionOfInterest | None = None,
        sampling: Settings.Sampling | None = None,
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

        if isinstance(color, zivid.settings2d.Settings2D) or color is None:
            self._color = color
        else:
            raise TypeError(
                "Unsupported type, expected: zivid.settings2d.Settings2D or None, got {value_type}".format(
                    value_type=type(color)
                )
            )

        if isinstance(engine, _zivid.Settings.Engine.enum) or engine is None:
            self._engine = _zivid.Settings.Engine(engine)
        elif isinstance(engine, str):
            self._engine = _zivid.Settings.Engine(self.Engine._valid_values[engine])
        else:
            raise TypeError("Unsupported type, expected: str or None, got {value_type}".format(value_type=type(engine)))

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

        if region_of_interest is None:
            region_of_interest = self.RegionOfInterest()
        if not isinstance(region_of_interest, self.RegionOfInterest):
            raise TypeError("Unsupported type: {value}".format(value=type(region_of_interest)))
        self._region_of_interest = region_of_interest

        if sampling is None:
            sampling = self.Sampling()
        if not isinstance(sampling, self.Sampling):
            raise TypeError("Unsupported type: {value}".format(value=type(sampling)))
        self._sampling = sampling

    @property
    def acquisitions(self):
        """List of Acquisition objects."""
        return self._acquisitions

    @property
    def color(self):
        """Specify the settings used for the 2D color image when doing a 2D+3D capture. The value type of this node is a `Zivid::Settings2D` object. This setting was introduced in SDK 2.14, as the recommended way to define the acquisition and processing settings for the 2D color image when doing 2D+3D captures. When this setting is set, it controls how the 2D color image of the 2D+3D capture is acquired and processed. This setting can be used to specify custom acquisition and processing settings that apply only to the 2D color image. These provided settings does not affect the 3D acquisition or processing. When this setting is not set, then the 2D color image is acquired based on `Settings/Acquisitions` and `Settings/Sampling/Color`, and processed based on `Settings/Processing/Color`. If `Settings/Sampling/Color` is set to `disabled`, then no 2D color image is acquired. This behavior is to be consistent with SDK 2.13 and earlier. In the next SDK major version, SDK 3.0, only this `Color` setting will control the 2D color image settings. The `Settings/Sampling/Color` and `Settings/Processing/Color` settings are deprecated as of 2.14, and will be removed from the API in SDK 3.0. Zivid recommends all users to use `Settings/Color` to define the 2D color image settings. Tip: If you want to convert an existing settings.yml file to use `Settings/Color`, you can import the .yml file into Zivid Studio and then re-export it to .yml."""
        return self._color

    @property
    def engine(self):
        """Set the Zivid Vision Engine to use. The Phase Engine is the fastest choice in terms of both acquisition time and total capture time, and is a good compromise between quality and speed. The Phase Engine is recommended for objects that are diffuse, opaque, and slightly specular, and is suitable for applications in logistics such as parcel induction. The Stripe Engine is built for exceptional point cloud quality in scenes with highly specular reflective objects. This makes the engine suitable for applications such as factory automation, manufacturing, and bin picking. Additional acquisition and processing time are required for the Stripe Engine. The Omni Engine is built for exceptional point cloud quality on all scenes, including scenes with extremely specular reflective objects, as well as transparent objects. This makes the Omni Engine suitable for applications such as piece picking. Same as for the Stripe Engine, it trades off speed for quality. The Omni Engine is only available for Zivid 2+ and Zivid 2+R. The Sage engine is built for use cases that require all points to be correct/accurate with particularly high confidence. This can be very suitable for eliminating problems such as reflection artifacts. This validation comes at the cost of speed, and sometimes reduced number of valid points due to the removal of low-confidence data. The Sage Engine is only available for Zivid 2+R."""
        if self._engine.value is None:
            return None
        for key, internal_value in self.Engine._valid_values.items():
            if internal_value == self._engine.value:
                return key
        raise ValueError("Unsupported value {value}".format(value=self._engine))

    @property
    def diagnostics(self):
        """When Diagnostics is enabled, additional diagnostic data is recorded during capture and included when saving the frame to a .zdf file. This enables Zivid's Customer Success team to provide better assistance and more thorough troubleshooting. Enabling Diagnostics increases the capture time and the RAM usage. It will also increase the size of the .zdf file. It is recommended to enable Diagnostics only when reporting issues to Zivid's support team."""
        return self._diagnostics

    @property
    def processing(self):
        """Settings related to processing of a capture, including filters and color balance."""
        return self._processing

    @property
    def region_of_interest(self):
        """Removes points outside the region of interest."""
        return self._region_of_interest

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

    @color.setter
    def color(self, value):
        if isinstance(value, zivid.settings2d.Settings2D) or value is None:
            self._color = value
        else:
            raise TypeError(
                "Unsupported type, expected: zivid.settings2d.Settings2D or None, got {value_type}".format(
                    value_type=type(value)
                )
            )

    @engine.setter
    def engine(self, value):
        if isinstance(value, str):
            self._engine = _zivid.Settings.Engine(self.Engine._valid_values[value])
        elif isinstance(value, _zivid.Settings.Engine.enum) or value is None:
            self._engine = _zivid.Settings.Engine(value)
        else:
            raise TypeError("Unsupported type, expected: str or None, got {value_type}".format(value_type=type(value)))

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

    @region_of_interest.setter
    def region_of_interest(self, value):
        if not isinstance(value, self.RegionOfInterest):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._region_of_interest = value

    @sampling.setter
    def sampling(self, value):
        if not isinstance(value, self.Sampling):
            raise TypeError("Unsupported type {value}".format(value=type(value)))
        self._sampling = value

    @classmethod
    def load(cls, file_name):
        return _to_settings(_zivid.Settings(str(file_name)))

    def save(self, file_name):
        _to_internal_settings(self).save(str(file_name))

    @classmethod
    def from_serialized(cls, value):
        return _to_settings(_zivid.Settings.from_serialized(str(value)))

    def serialize(self):
        return _to_internal_settings(self).serialize()

    def __eq__(self, other):
        if (
            self._acquisitions == other._acquisitions
            and self._color == other._color
            and self._engine == other._engine
            and self._diagnostics == other._diagnostics
            and self._processing == other._processing
            and self._region_of_interest == other._region_of_interest
            and self._sampling == other._sampling
        ):
            return True
        return False

    def __str__(self):
        return str(_to_internal_settings(self))

    def __deepcopy__(self, memodict):
        # Create deep copy by converting to internal representation and back.
        # memodict not used since conversion creates entirely new objects.
        return _to_settings(_to_internal_settings(self))


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


def _to_settings_processing_color_balance(internal_balance):
    return Settings.Processing.Color.Balance(
        blue=internal_balance.blue.value,
        green=internal_balance.green.value,
        red=internal_balance.red.value,
    )


def _to_settings_processing_color_experimental(internal_experimental):
    return Settings.Processing.Color.Experimental(
        mode=internal_experimental.mode.value,
    )


def _to_settings_processing_color(internal_color):
    return Settings.Processing.Color(
        balance=_to_settings_processing_color_balance(internal_color.balance),
        experimental=_to_settings_processing_color_experimental(internal_color.experimental),
        gamma=internal_color.gamma.value,
    )


def _to_settings_processing_filters_cluster_removal(internal_removal):
    return Settings.Processing.Filters.Cluster.Removal(
        enabled=internal_removal.enabled.value,
        max_neighbor_distance=internal_removal.max_neighbor_distance.value,
        min_area=internal_removal.min_area.value,
    )


def _to_settings_processing_filters_cluster(internal_cluster):
    return Settings.Processing.Filters.Cluster(
        removal=_to_settings_processing_filters_cluster_removal(internal_cluster.removal),
    )


def _to_settings_processing_filters_experimental_contrast_distortion_correction(internal_correction):
    return Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
        enabled=internal_correction.enabled.value,
        strength=internal_correction.strength.value,
    )


def _to_settings_processing_filters_experimental_contrast_distortion_removal(internal_removal):
    return Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_experimental_contrast_distortion(internal_contrast_distortion):
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


def _to_settings_processing_filters_hole_repair(internal_repair):
    return Settings.Processing.Filters.Hole.Repair(
        enabled=internal_repair.enabled.value,
        hole_size=internal_repair.hole_size.value,
        strictness=internal_repair.strictness.value,
    )


def _to_settings_processing_filters_hole(internal_hole):
    return Settings.Processing.Filters.Hole(
        repair=_to_settings_processing_filters_hole_repair(internal_hole.repair),
    )


def _to_settings_processing_filters_noise_removal(internal_removal):
    return Settings.Processing.Filters.Noise.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_noise_repair(internal_repair):
    return Settings.Processing.Filters.Noise.Repair(
        enabled=internal_repair.enabled.value,
    )


def _to_settings_processing_filters_noise_suppression(internal_suppression):
    return Settings.Processing.Filters.Noise.Suppression(
        enabled=internal_suppression.enabled.value,
    )


def _to_settings_processing_filters_noise(internal_noise):
    return Settings.Processing.Filters.Noise(
        removal=_to_settings_processing_filters_noise_removal(internal_noise.removal),
        repair=_to_settings_processing_filters_noise_repair(internal_noise.repair),
        suppression=_to_settings_processing_filters_noise_suppression(internal_noise.suppression),
    )


def _to_settings_processing_filters_outlier_removal(internal_removal):
    return Settings.Processing.Filters.Outlier.Removal(
        enabled=internal_removal.enabled.value,
        threshold=internal_removal.threshold.value,
    )


def _to_settings_processing_filters_outlier(internal_outlier):
    return Settings.Processing.Filters.Outlier(
        removal=_to_settings_processing_filters_outlier_removal(internal_outlier.removal),
    )


def _to_settings_processing_filters_reflection_removal(internal_removal):
    return Settings.Processing.Filters.Reflection.Removal(
        enabled=internal_removal.enabled.value,
        mode=internal_removal.mode.value,
    )


def _to_settings_processing_filters_reflection(internal_reflection):
    return Settings.Processing.Filters.Reflection(
        removal=_to_settings_processing_filters_reflection_removal(internal_reflection.removal),
    )


def _to_settings_processing_filters_smoothing_gaussian(internal_gaussian):
    return Settings.Processing.Filters.Smoothing.Gaussian(
        enabled=internal_gaussian.enabled.value,
        sigma=internal_gaussian.sigma.value,
    )


def _to_settings_processing_filters_smoothing(internal_smoothing):
    return Settings.Processing.Filters.Smoothing(
        gaussian=_to_settings_processing_filters_smoothing_gaussian(internal_smoothing.gaussian),
    )


def _to_settings_processing_filters(internal_filters):
    return Settings.Processing.Filters(
        cluster=_to_settings_processing_filters_cluster(internal_filters.cluster),
        experimental=_to_settings_processing_filters_experimental(internal_filters.experimental),
        hole=_to_settings_processing_filters_hole(internal_filters.hole),
        noise=_to_settings_processing_filters_noise(internal_filters.noise),
        outlier=_to_settings_processing_filters_outlier(internal_filters.outlier),
        reflection=_to_settings_processing_filters_reflection(internal_filters.reflection),
        smoothing=_to_settings_processing_filters_smoothing(internal_filters.smoothing),
    )


def _to_settings_processing_resampling(internal_resampling):
    return Settings.Processing.Resampling(
        mode=internal_resampling.mode.value,
    )


def _to_settings_processing(internal_processing):
    return Settings.Processing(
        color=_to_settings_processing_color(internal_processing.color),
        filters=_to_settings_processing_filters(internal_processing.filters),
        resampling=_to_settings_processing_resampling(internal_processing.resampling),
    )


def _to_settings_region_of_interest_box(internal_box):
    return Settings.RegionOfInterest.Box(
        enabled=internal_box.enabled.value,
        extents=internal_box.extents.value,
        point_a=internal_box.point_a.value,
        point_b=internal_box.point_b.value,
        point_o=internal_box.point_o.value,
    )


def _to_settings_region_of_interest_depth(internal_depth):
    return Settings.RegionOfInterest.Depth(
        enabled=internal_depth.enabled.value,
        range=internal_depth.range.value,
    )


def _to_settings_region_of_interest(internal_region_of_interest):
    return Settings.RegionOfInterest(
        box=_to_settings_region_of_interest_box(internal_region_of_interest.box),
        depth=_to_settings_region_of_interest_depth(internal_region_of_interest.depth),
    )


def _to_settings_sampling(internal_sampling):
    return Settings.Sampling(
        color=internal_sampling.color.value,
        pixel=internal_sampling.pixel.value,
    )


def _to_settings(internal_settings):
    return Settings(
        acquisitions=[_to_settings_acquisition(value) for value in internal_settings.acquisitions.value],
        diagnostics=_to_settings_diagnostics(internal_settings.diagnostics),
        processing=_to_settings_processing(internal_settings.processing),
        region_of_interest=_to_settings_region_of_interest(internal_settings.region_of_interest),
        sampling=_to_settings_sampling(internal_settings.sampling),
        color=(
            zivid.settings2d._to_settings2d(internal_settings.color.value)
            if internal_settings.color.value is not None
            else None
        ),
        engine=internal_settings.engine.value,
    )


def _to_internal_settings_acquisition(acquisition):
    internal_acquisition = _zivid.Settings.Acquisition()

    internal_acquisition.aperture = _zivid.Settings.Acquisition.Aperture(acquisition.aperture)
    internal_acquisition.brightness = _zivid.Settings.Acquisition.Brightness(acquisition.brightness)
    internal_acquisition.exposure_time = _zivid.Settings.Acquisition.ExposureTime(acquisition.exposure_time)
    internal_acquisition.gain = _zivid.Settings.Acquisition.Gain(acquisition.gain)

    return internal_acquisition


def _to_internal_settings_diagnostics(diagnostics):
    internal_diagnostics = _zivid.Settings.Diagnostics()

    internal_diagnostics.enabled = _zivid.Settings.Diagnostics.Enabled(diagnostics.enabled)

    return internal_diagnostics


def _to_internal_settings_processing_color_balance(balance):
    internal_balance = _zivid.Settings.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings.Processing.Color.Balance.Blue(balance.blue)
    internal_balance.green = _zivid.Settings.Processing.Color.Balance.Green(balance.green)
    internal_balance.red = _zivid.Settings.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def _to_internal_settings_processing_color_experimental(experimental):
    internal_experimental = _zivid.Settings.Processing.Color.Experimental()

    internal_experimental.mode = _zivid.Settings.Processing.Color.Experimental.Mode(experimental._mode.value)

    return internal_experimental


def _to_internal_settings_processing_color(color):
    internal_color = _zivid.Settings.Processing.Color()

    internal_color.gamma = _zivid.Settings.Processing.Color.Gamma(color.gamma)

    internal_color.balance = _to_internal_settings_processing_color_balance(color.balance)
    internal_color.experimental = _to_internal_settings_processing_color_experimental(color.experimental)
    return internal_color


def _to_internal_settings_processing_filters_cluster_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Cluster.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Cluster.Removal.Enabled(removal.enabled)
    internal_removal.max_neighbor_distance = _zivid.Settings.Processing.Filters.Cluster.Removal.MaxNeighborDistance(
        removal.max_neighbor_distance
    )
    internal_removal.min_area = _zivid.Settings.Processing.Filters.Cluster.Removal.MinArea(removal.min_area)

    return internal_removal


def _to_internal_settings_processing_filters_cluster(cluster):
    internal_cluster = _zivid.Settings.Processing.Filters.Cluster()

    internal_cluster.removal = _to_internal_settings_processing_filters_cluster_removal(cluster.removal)
    return internal_cluster


def _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(correction):
    internal_correction = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction()

    internal_correction.enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Enabled(
        correction.enabled
    )
    internal_correction.strength = (
        _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction.Strength(correction.strength)
    )

    return internal_correction


def _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Enabled(
        removal.enabled
    )
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal.Threshold(
        removal.threshold
    )

    return internal_removal


def _to_internal_settings_processing_filters_experimental_contrast_distortion(contrast_distortion):
    internal_contrast_distortion = _zivid.Settings.Processing.Filters.Experimental.ContrastDistortion()

    internal_contrast_distortion.correction = (
        _to_internal_settings_processing_filters_experimental_contrast_distortion_correction(
            contrast_distortion.correction
        )
    )
    internal_contrast_distortion.removal = (
        _to_internal_settings_processing_filters_experimental_contrast_distortion_removal(contrast_distortion.removal)
    )
    return internal_contrast_distortion


def _to_internal_settings_processing_filters_experimental(experimental):
    internal_experimental = _zivid.Settings.Processing.Filters.Experimental()

    internal_experimental.contrast_distortion = (
        _to_internal_settings_processing_filters_experimental_contrast_distortion(experimental.contrast_distortion)
    )
    return internal_experimental


def _to_internal_settings_processing_filters_hole_repair(repair):
    internal_repair = _zivid.Settings.Processing.Filters.Hole.Repair()

    internal_repair.enabled = _zivid.Settings.Processing.Filters.Hole.Repair.Enabled(repair.enabled)
    internal_repair.hole_size = _zivid.Settings.Processing.Filters.Hole.Repair.HoleSize(repair.hole_size)
    internal_repair.strictness = _zivid.Settings.Processing.Filters.Hole.Repair.Strictness(repair.strictness)

    return internal_repair


def _to_internal_settings_processing_filters_hole(hole):
    internal_hole = _zivid.Settings.Processing.Filters.Hole()

    internal_hole.repair = _to_internal_settings_processing_filters_hole_repair(hole.repair)
    return internal_hole


def _to_internal_settings_processing_filters_noise_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Noise.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Noise.Removal.Enabled(removal.enabled)
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Noise.Removal.Threshold(removal.threshold)

    return internal_removal


def _to_internal_settings_processing_filters_noise_repair(repair):
    internal_repair = _zivid.Settings.Processing.Filters.Noise.Repair()

    internal_repair.enabled = _zivid.Settings.Processing.Filters.Noise.Repair.Enabled(repair.enabled)

    return internal_repair


def _to_internal_settings_processing_filters_noise_suppression(suppression):
    internal_suppression = _zivid.Settings.Processing.Filters.Noise.Suppression()

    internal_suppression.enabled = _zivid.Settings.Processing.Filters.Noise.Suppression.Enabled(suppression.enabled)

    return internal_suppression


def _to_internal_settings_processing_filters_noise(noise):
    internal_noise = _zivid.Settings.Processing.Filters.Noise()

    internal_noise.removal = _to_internal_settings_processing_filters_noise_removal(noise.removal)
    internal_noise.repair = _to_internal_settings_processing_filters_noise_repair(noise.repair)
    internal_noise.suppression = _to_internal_settings_processing_filters_noise_suppression(noise.suppression)
    return internal_noise


def _to_internal_settings_processing_filters_outlier_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Outlier.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Outlier.Removal.Enabled(removal.enabled)
    internal_removal.threshold = _zivid.Settings.Processing.Filters.Outlier.Removal.Threshold(removal.threshold)

    return internal_removal


def _to_internal_settings_processing_filters_outlier(outlier):
    internal_outlier = _zivid.Settings.Processing.Filters.Outlier()

    internal_outlier.removal = _to_internal_settings_processing_filters_outlier_removal(outlier.removal)
    return internal_outlier


def _to_internal_settings_processing_filters_reflection_removal(removal):
    internal_removal = _zivid.Settings.Processing.Filters.Reflection.Removal()

    internal_removal.enabled = _zivid.Settings.Processing.Filters.Reflection.Removal.Enabled(removal.enabled)
    internal_removal.mode = _zivid.Settings.Processing.Filters.Reflection.Removal.Mode(removal._mode.value)

    return internal_removal


def _to_internal_settings_processing_filters_reflection(reflection):
    internal_reflection = _zivid.Settings.Processing.Filters.Reflection()

    internal_reflection.removal = _to_internal_settings_processing_filters_reflection_removal(reflection.removal)
    return internal_reflection


def _to_internal_settings_processing_filters_smoothing_gaussian(gaussian):
    internal_gaussian = _zivid.Settings.Processing.Filters.Smoothing.Gaussian()

    internal_gaussian.enabled = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Enabled(gaussian.enabled)
    internal_gaussian.sigma = _zivid.Settings.Processing.Filters.Smoothing.Gaussian.Sigma(gaussian.sigma)

    return internal_gaussian


def _to_internal_settings_processing_filters_smoothing(smoothing):
    internal_smoothing = _zivid.Settings.Processing.Filters.Smoothing()

    internal_smoothing.gaussian = _to_internal_settings_processing_filters_smoothing_gaussian(smoothing.gaussian)
    return internal_smoothing


def _to_internal_settings_processing_filters(filters):
    internal_filters = _zivid.Settings.Processing.Filters()

    internal_filters.cluster = _to_internal_settings_processing_filters_cluster(filters.cluster)
    internal_filters.experimental = _to_internal_settings_processing_filters_experimental(filters.experimental)
    internal_filters.hole = _to_internal_settings_processing_filters_hole(filters.hole)
    internal_filters.noise = _to_internal_settings_processing_filters_noise(filters.noise)
    internal_filters.outlier = _to_internal_settings_processing_filters_outlier(filters.outlier)
    internal_filters.reflection = _to_internal_settings_processing_filters_reflection(filters.reflection)
    internal_filters.smoothing = _to_internal_settings_processing_filters_smoothing(filters.smoothing)
    return internal_filters


def _to_internal_settings_processing_resampling(resampling):
    internal_resampling = _zivid.Settings.Processing.Resampling()

    internal_resampling.mode = _zivid.Settings.Processing.Resampling.Mode(resampling._mode.value)

    return internal_resampling


def _to_internal_settings_processing(processing):
    internal_processing = _zivid.Settings.Processing()

    internal_processing.color = _to_internal_settings_processing_color(processing.color)
    internal_processing.filters = _to_internal_settings_processing_filters(processing.filters)
    internal_processing.resampling = _to_internal_settings_processing_resampling(processing.resampling)
    return internal_processing


def _to_internal_settings_region_of_interest_box(box):
    internal_box = _zivid.Settings.RegionOfInterest.Box()

    internal_box.enabled = _zivid.Settings.RegionOfInterest.Box.Enabled(box.enabled)
    internal_box.extents = _zivid.Settings.RegionOfInterest.Box.Extents(box.extents)
    internal_box.point_a = _zivid.Settings.RegionOfInterest.Box.PointA(box.point_a)
    internal_box.point_b = _zivid.Settings.RegionOfInterest.Box.PointB(box.point_b)
    internal_box.point_o = _zivid.Settings.RegionOfInterest.Box.PointO(box.point_o)

    return internal_box


def _to_internal_settings_region_of_interest_depth(depth):
    internal_depth = _zivid.Settings.RegionOfInterest.Depth()

    internal_depth.enabled = _zivid.Settings.RegionOfInterest.Depth.Enabled(depth.enabled)
    internal_depth.range = _zivid.Settings.RegionOfInterest.Depth.Range(depth.range)

    return internal_depth


def _to_internal_settings_region_of_interest(region_of_interest):
    internal_region_of_interest = _zivid.Settings.RegionOfInterest()

    internal_region_of_interest.box = _to_internal_settings_region_of_interest_box(region_of_interest.box)
    internal_region_of_interest.depth = _to_internal_settings_region_of_interest_depth(region_of_interest.depth)
    return internal_region_of_interest


def _to_internal_settings_sampling(sampling):
    internal_sampling = _zivid.Settings.Sampling()

    internal_sampling.color = _zivid.Settings.Sampling.Color(sampling._color.value)
    internal_sampling.pixel = _zivid.Settings.Sampling.Pixel(sampling._pixel.value)

    return internal_sampling


def _to_internal_settings(settings):
    internal_settings = _zivid.Settings()

    temp_acquisitions = _zivid.Settings.Acquisitions()
    for value in settings.acquisitions:
        temp_acquisitions.append(_to_internal_settings_acquisition(value))
    internal_settings.acquisitions = temp_acquisitions

    internal_settings.color = _zivid.Settings.Color(
        zivid.settings2d._to_internal_settings2d(settings.color) if settings.color is not None else None
    )
    internal_settings.engine = _zivid.Settings.Engine(settings._engine.value)

    internal_settings.diagnostics = _to_internal_settings_diagnostics(settings.diagnostics)
    internal_settings.processing = _to_internal_settings_processing(settings.processing)
    internal_settings.region_of_interest = _to_internal_settings_region_of_interest(settings.region_of_interest)
    internal_settings.sampling = _to_internal_settings_sampling(settings.sampling)
    return internal_settings
