import pytest


def test_available(file_camera):
    available = file_camera.state.available
    assert available is not None
    assert isinstance(available, bool)


def test_connected(file_camera):
    connected = file_camera.state.connected
    assert connected is not None
    assert isinstance(connected, bool)


def test_temperature(file_camera):
    from zivid.camera_state import CameraState

    temperature = file_camera.state.temperature
    assert temperature is not None
    assert isinstance(temperature, CameraState.Temperature)


def test_temperature_dmd(file_camera):
    import numbers

    dmd = file_camera.state.temperature.dmd
    assert dmd is not None
    assert isinstance(dmd, numbers.Real)


def test_temperature_general(file_camera):
    import numbers

    general = file_camera.state.temperature.general
    assert general is not None
    assert isinstance(general, numbers.Real)


def test_temperature_led(file_camera):
    import numbers

    led = file_camera.state.temperature.led
    assert led is not None
    assert isinstance(led, numbers.Real)


def test_temperature_lens(file_camera):
    import numbers

    lens = file_camera.state.temperature.lens
    assert lens is not None
    assert isinstance(lens, numbers.Real)


def test_temperature_pcb(file_camera):
    import numbers

    pcb = file_camera.state.temperature.pcb
    assert pcb is not None
    assert isinstance(pcb, numbers.Real)


def test_illegal_set_state(file_camera):

    with pytest.raises(AttributeError):
        file_camera.state = file_camera.state


def test_equal_state():
    from zivid.camera_state import CameraState

    state1 = CameraState(available=True)
    state2 = CameraState(available=True)

    assert state1 == state2


def test_not_equal_state():
    from zivid.camera_state import CameraState

    state1 = CameraState(available=True)
    state2 = CameraState(available=False)

    assert state1 != state2


def test_equal_temperature():
    from zivid.camera_state import CameraState

    temperature1 = CameraState.Temperature(dmd=33)
    temperature2 = CameraState.Temperature(dmd=33)

    assert temperature1 == temperature2


def test_not_equal_temperature():
    from zivid.camera_state import CameraState

    temperature1 = CameraState.Temperature(dmd=33)
    temperature2 = CameraState.Temperature(dmd=44)

    assert temperature1 != temperature2
