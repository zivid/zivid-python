import pytest


def test_available(shared_file_camera):
    available = shared_file_camera.state.available
    assert available is not None
    assert isinstance(available, bool)


def test_connected(shared_file_camera):
    connected = shared_file_camera.state.connected
    assert connected is not None
    assert isinstance(connected, bool)


def test_temperature(shared_file_camera):
    from zivid.camera_state import CameraState

    temperature = shared_file_camera.state.temperature
    assert temperature is not None
    assert isinstance(temperature, CameraState.Temperature)


def test_temperature_dmd(shared_file_camera):
    import numbers

    dmd = shared_file_camera.state.temperature.dmd
    assert dmd is not None
    assert isinstance(dmd, numbers.Real)


def test_temperature_general(shared_file_camera):
    import numbers

    general = shared_file_camera.state.temperature.general
    assert general is not None
    assert isinstance(general, numbers.Real)


def test_temperature_led(shared_file_camera):
    import numbers

    led = shared_file_camera.state.temperature.led
    assert led is not None
    assert isinstance(led, numbers.Real)


def test_temperature_lens(shared_file_camera):
    import numbers

    lens = shared_file_camera.state.temperature.lens
    assert lens is not None
    assert isinstance(lens, numbers.Real)


def test_temperature_pcb(shared_file_camera):
    import numbers

    pcb = shared_file_camera.state.temperature.pcb
    assert pcb is not None
    assert isinstance(pcb, numbers.Real)


def test_illegal_set_state(shared_file_camera):
    with pytest.raises(AttributeError):
        shared_file_camera.state = shared_file_camera.state


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
