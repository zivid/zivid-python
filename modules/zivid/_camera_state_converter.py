"""Auto generated, do not edit."""
import zivid
import _zivid


def to_camera_state_temperature(internal_temperature):
    return zivid.CameraState.Temperature(
        dmd=internal_temperature.dmd.value,
        general=internal_temperature.general.value,
        led=internal_temperature.led.value,
        lens=internal_temperature.lens.value,
        pcb=internal_temperature.pcb.value,
    )


def to_camera_state(internal_camera_state):
    return zivid.CameraState(
        temperature=to_camera_state_temperature(internal_camera_state.temperature),
        available=internal_camera_state.available.value,
        connected=internal_camera_state.connected.value,
    )


def to_internal_camera_state_temperature(temperature):
    internal_temperature = _zivid.CameraState.Temperature()

    internal_temperature.dmd = _zivid.CameraState.Temperature.DMD(temperature.dmd)
    internal_temperature.general = _zivid.CameraState.Temperature.General(
        temperature.general
    )
    internal_temperature.led = _zivid.CameraState.Temperature.LED(temperature.led)
    internal_temperature.lens = _zivid.CameraState.Temperature.Lens(temperature.lens)
    internal_temperature.pcb = _zivid.CameraState.Temperature.PCB(temperature.pcb)

    return internal_temperature


def to_internal_camera_state(camera_state):
    internal_camera_state = _zivid.CameraState()

    internal_camera_state.available = _zivid.CameraState.Available(
        camera_state.available
    )
    internal_camera_state.connected = _zivid.CameraState.Connected(
        camera_state.connected
    )

    internal_camera_state.temperature = to_internal_camera_state_temperature(
        camera_state.temperature
    )
    return internal_camera_state
