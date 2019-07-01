"""Contains CameraState class."""
import _zivid


class CameraState:  # pylint: disable=too-many-instance-attributes, too-few-public-methods
    """Information about camera connection state, live mode, temperatures, etc."""

    class Temperature:  # pylint: disable=too-few-public-methods
        """Current temperature(s)."""

        def __init__(  # pylint: disable=too-many-arguments
            self,
            dmd=_zivid.CameraState.Temperature.DMD().value,
            general=_zivid.CameraState.Temperature.General().value,
            led=_zivid.CameraState.Temperature.LED().value,
            lens=_zivid.CameraState.Temperature.Lens().value,
            pcb=_zivid.CameraState.Temperature.PCB().value,
        ):
            """Initialize temperature.

            Args:
                dmd: a real number
                general: a real number
                led: a real number
                lens: a real number
                pcb: a real number

            """
            self.dmd = dmd
            self.general = general
            self.led = led
            self.lens = lens
            self.pcb = pcb

        def __eq__(self, other):
            return (
                self.dmd == other.dmd
                and self.general == other.general
                and self.led == other.led
                and self.lens == other.lens
                and self.pcb == other.pcb
            )

    def __init__(
        self,
        available=_zivid.CameraState.Available().value,
        connected=_zivid.CameraState.Connected().value,
        live=_zivid.CameraState.Live().value,
        temperature=Temperature(),
    ):
        """Initialize camera state.

        Args:
            available: a bool
            connected: a bool
            live: a bool
            temperature: a temperature instance

        """
        self.available = available
        self.connected = connected
        self.live = live
        self.temperature = temperature

    def __eq__(self, other):
        return (
            self.available == other.available
            and self.connected == other.connected
            and self.live == other.live
            and self.temperature == other.temperature
        )
