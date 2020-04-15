import zivid


def to_camera_state(internal_camera_state):
    def _to_temperature(internal_temperature):

        return zivid.CameraState.Temperature(
            dmd=internal_temperature.dmd.value,
            general=internal_temperature.general.value,
            led=internal_temperature.led.value,
            lens=internal_temperature.lens.value,
            pcb=internal_temperature.pcb.value,
        )

    global to_temperature
    to_temperature = _to_temperature
    return zivid.CameraState(
        temperature=_to_temperature(internal_camera_state.temperature),
        available=internal_camera_state.available.value,
        connected=internal_camera_state.connected.value,
    )
