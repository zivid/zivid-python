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


def test_network_file_camera(shared_file_camera):
    from zivid.camera_state import CameraState

    assert isinstance(shared_file_camera.state.network, CameraState.Network)
    assert isinstance(shared_file_camera.state.network.ipv4, CameraState.Network.IPV4)
    assert isinstance(shared_file_camera.state.network.ipv4.address, str)
    assert shared_file_camera.state.network.ipv4.address == ""

    assert isinstance(shared_file_camera.state.network.local_interfaces, list)
    assert len(shared_file_camera.state.network.local_interfaces) == 0


@pytest.mark.physical_camera
def test_network(physical_camera):
    from zivid.camera_state import CameraState

    assert isinstance(physical_camera.state.network, CameraState.Network)
    assert isinstance(physical_camera.state.network.ipv4, CameraState.Network.IPV4)
    assert isinstance(physical_camera.state.network.ipv4.address, str)
    assert physical_camera.state.network.ipv4.address != ""

    assert isinstance(physical_camera.state.network.local_interfaces, list)
    assert all(
        isinstance(interface, CameraState.Network.LocalInterface)
        for interface in physical_camera.state.network.local_interfaces
    )

    for interface in physical_camera.state.network.local_interfaces:
        assert isinstance(interface.interface_name, str)
        assert interface.interface_name != ""
        assert isinstance(interface.ipv4, CameraState.Network.LocalInterface.IPV4)
        assert isinstance(interface.ipv4.subnets, list)
        assert all(
            isinstance(subnet, CameraState.Network.LocalInterface.IPV4.Subnet)
            for subnet in interface.ipv4.subnets
        )

        for subnet in interface.ipv4.subnets:
            assert isinstance(subnet.address, str)
            assert subnet.address != ""
            assert isinstance(subnet.mask, str)
            assert subnet.mask != ""
