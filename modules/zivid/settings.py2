"""Contains Settings class."""
import _zivid


class Settings:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Settings for a Zivid camera."""

    class Filters:  # pylint: disable=too-few-public-methods
        """Toggle on or off various filters."""

        class Contrast:  # pylint: disable=too-few-public-methods
            """Discard points with low contrast values."""

            def __init__(
                self,
                enabled=_zivid.Settings().filters.contrast.enabled.value,
                threshold=_zivid.Settings().filters.contrast.threshold.value,
            ):
                """Initialize contrast filter.

                Args:
                    enabled: a bool
                    threshold: a real number

                """

                self.enabled = enabled
                self.threshold = threshold

            def __eq__(self, other):
                if self.enabled == other.enabled and self.threshold == other.threshold:
                    return True
                return False

            def __str__(self):
                return """Contrast:
enabled: {}
threshold: {}""".format(
                    self.enabled, self.threshold
                )

        class Outlier:  # pylint: disable=too-few-public-methods
            """Outlier filter based on connected components."""

            def __init__(
                self,
                enabled=_zivid.Settings().filters.outlier.enabled.value,
                threshold=_zivid.Settings().filters.outlier.threshold.value,
            ):
                """Initialize outlier filter.

                Args:
                    enabled: a bool
                    threshold: a real number

                """
                self.enabled = enabled
                self.threshold = threshold

            def __eq__(self, other):
                if self.enabled == other.enabled and self.threshold == other.threshold:
                    return True
                return False

            def __str__(self):
                return """Outlier:
enabled: {}
threshold: {}""".format(
                    self.enabled, self.threshold
                )

        class Saturated:  # pylint: disable=too-few-public-methods
            """Discard pixels that are saturated in the image."""

            def __init__(
                self, enabled=_zivid.Settings().filters.saturated.enabled.value
            ):
                """Initialize saturated filter.

                Args:
                    enabled: a bool

                """
                self.enabled = enabled

            def __eq__(self, other):
                if self.enabled == other.enabled:
                    return True
                return False

            def __str__(self):
                return """Saturated:
enabled: {}""".format(
                    self.enabled
                )

        class Reflection:  # pylint: disable=too-few-public-methods
            """Represents camera reflection filter."""

            def __init__(
                self, enabled=_zivid.Settings().filters.reflection.enabled.value
            ):
                """Initialize reflection filter.

                Args:
                    enabled: a bool

                """
                self.enabled = enabled

            def __eq__(self, other):
                if self.enabled == other.enabled:
                    return True
                return False

            def __str__(self):
                return """Reflection:
enabled: {}""".format(
                    self.enabled
                )

        class Gaussian:  # pylint: disable=too-few-public-methods
            """Gaussian smoothing of the point cloud."""

            def __init__(
                self,
                enabled=_zivid.Settings().filters.gaussian.enabled.value,
                sigma=_zivid.Settings().filters.gaussian.sigma.value,
            ):
                """Initialize gaussian filter.

                Args:
                    enabled: a bool
                    sigma: a real number

                """
                self.enabled = enabled
                self.sigma = sigma

            def __eq__(self, other):
                if self.enabled == other.enabled and self.sigma == other.sigma:
                    return True
                return False

            def __str__(self):
                return """Gaussian:
enabled: {}
sigma: {}""".format(
                    self.enabled, self.sigma
                )

        def __init__(  # pylint: disable=too-many-arguments
            self,
            contrast=Contrast(),
            outlier=Outlier(),
            saturated=Saturated(),
            reflection=Reflection(),
            gaussian=Gaussian(),
        ):
            """Initialize filters.

            Args:
                contrast: a contrast filter object
                outlier: a outlier filter object
                saturated: a saturated filter object
                reflection: a reflection filter object
                gaussian: a gaussian filter object

            """
            self.contrast = contrast
            self.outlier = outlier
            self.saturated = saturated
            self.reflection = reflection
            self.gaussian = gaussian

        def __eq__(self, other):
            if (
                self.contrast == other.contrast
                and self.outlier == other.outlier
                and self.saturated == other.saturated
                and self.reflection == other.reflection
                and self.gaussian == other.gaussian
            ):
                return True
            return False

        def __str__(self):
            return """Filters:
contrast: {}
outlier: {}
saturated: {}
reflection: {}
gaussian: {}""".format(
                self.contrast,
                self.outlier,
                self.saturated,
                self.reflection,
                self.gaussian,
            )

    def __init__(  # pylint: disable=too-many-arguments
        self,
        bidirectional=_zivid.Settings().bidirectional.value,
        blue_balance=_zivid.Settings().bluebalance.value,
        brightness=_zivid.Settings().brightness.value,
        exposure_time=_zivid.Settings().exposuretime.value,
        filters=Filters(),
        gain=_zivid.Settings().gain.value,
        iris=_zivid.Settings().iris.value,
        red_balance=_zivid.Settings().redbalance.value,
    ):
        """Initialize saturated filter.

        Args:
            bidirectional: a bool
            blue_balance: a real number
            brightness: a real number
            exposure_time: a datetime.timedelta object
            filters: a filters object
            gain: a real number
            iris: an int
            red_balance: a real number

        """
        self.bidirectional = bidirectional
        self.blue_balance = blue_balance
        self.brightness = brightness
        self.exposure_time = exposure_time
        self.filters = filters
        self.gain = gain
        self.iris = iris
        self.red_balance = red_balance

    def __eq__(self, other):
        if (
            self.bidirectional  # pylint: disable=too-many-boolean-expressions
            == other.bidirectional
            and self.blue_balance == other.blue_balance
            and self.brightness == other.brightness
            and self.exposure_time == other.exposure_time
            and self.filters == other.filters
            and self.gain == other.gain
            and self.iris == other.iris
            and self.red_balance == other.red_balance
        ):
            return True
        return False

    def __str__(self):
        return """Settings:
bidirectional: {}
blue_balance: {}
brightness: {}
exposure_time: {}
filters: {}
gain: {}
iris: {}
red_balance: {}""".format(
            self.bidirectional,
            self.blue_balance,
            self.brightness,
            self.exposure_time,
            self.filters,
            self.gain,
            self.iris,
            self.red_balance,
        )
