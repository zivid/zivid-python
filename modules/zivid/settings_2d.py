"""Contains Settings2D class."""
import _zivid


class Settings2D:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """2D Settings for a Zivid camera."""

    def __init__(
        self,
        brightness=_zivid.Settings2D().brightness.value,
        exposure_time=_zivid.Settings2D().exposuretime.value,
        gain=_zivid.Settings2D().gain.value,
        iris=_zivid.Settings2D().iris.value,
    ):
        """Initialize Settings2D.

        Args:
            brightness: a real number
            exposure_time: a datetime.timedelta object
            gain: a real number
            iris: an int

        """
        self.brightness = brightness
        self.exposure_time = exposure_time
        self.gain = gain
        self.iris = iris

    def __eq__(self, other):
        if (
            self.brightness == other.brightness
            and self.exposure_time == other.exposure_time
            and self.gain == other.gain
            and self.iris == other.iris
        ):
            return True
        return False

    def __str__(self):
        return """Settings2D:
brightness: {}
exposure_time: {}
gain: {}
iris: {}""".format(
            self.brightness, self.exposure_time, self.gain, self.iris
        )
