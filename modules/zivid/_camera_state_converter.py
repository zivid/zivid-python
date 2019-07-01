"""Contains functions for converting camera state to internal camera state."""
from zivid.camera_state import CameraState


def to_camera_state(internal_camera_state):
    """Convert camera state to internal camera state.

    Args:
        internal_camera_state: a internal camera state object

    Returns:
        a camera state object

    """

    def to_temperature(internal_temperature):
        return CameraState.Temperature(
            dmd=internal_temperature.dmd.value,
            general=internal_temperature.general.value,
            led=internal_temperature.led.value,
            lens=internal_temperature.lens.value,
            pcb=internal_temperature.pcb.value,
        )

    return CameraState(
        available=internal_camera_state.available.value,
        connected=internal_camera_state.connected.value,
        live=internal_camera_state.live.value,
        temperature=to_temperature(internal_camera_state.temperature),
    )
