import pytest
import zivid


class ScopeExit:
    def __init__(self, exit_func):
        self.exit_func = exit_func

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_func()


class DisconnectCamera(ScopeExit):
    def __init__(self, camera):
        camera.disconnect()
        super().__init__(lambda: camera.connect)


class RestoreNetworkConfiguration(ScopeExit):
    def __init__(self, camera):
        network_configuration = camera.network_configuration
        super().__init__(
            lambda: camera.apply_network_configuration(network_configuration)
        )


def test_default_network_configuration():
    network_configuration = zivid.NetworkConfiguration()
    assert (
        network_configuration.ipv4.mode == zivid.NetworkConfiguration.IPV4.Mode.manual
    )
    assert network_configuration.ipv4.address == "172.28.60.5"
    assert network_configuration.ipv4.subnet_mask == "255.255.255.0"


@pytest.mark.physical_camera
def test_fetch_network_configuration_while_connected(physical_camera):
    network_configuration = physical_camera.network_configuration
    assert network_configuration is not None
    assert isinstance(network_configuration, zivid.NetworkConfiguration)
    assert (
        network_configuration.ipv4.mode
        in zivid.NetworkConfiguration.IPV4.Mode.valid_values()
    )
    assert network_configuration.ipv4.address
    assert network_configuration.ipv4.subnet_mask


@pytest.mark.physical_camera
def test_fetch_network_configuration_while_not_connected(physical_camera):
    with DisconnectCamera(physical_camera):
        network_configuration = physical_camera.network_configuration
        assert network_configuration is not None
        assert isinstance(network_configuration, zivid.NetworkConfiguration)
        assert (
            network_configuration.ipv4.mode
            in zivid.NetworkConfiguration.IPV4.Mode.valid_values()
        )
        assert network_configuration.ipv4.address
        assert network_configuration.ipv4.subnet_mask


@pytest.mark.physical_camera
def test_apply_network_configuration_fails_while_connected(physical_camera):
    with pytest.raises(RuntimeError):
        physical_camera.apply_network_configuration(zivid.NetworkConfiguration())


@pytest.mark.physical_camera
def test_apply_default_network_configuration(physical_camera):
    with DisconnectCamera(physical_camera):
        with RestoreNetworkConfiguration(physical_camera):
            physical_camera.apply_network_configuration(zivid.NetworkConfiguration())
            assert physical_camera.network_configuration == zivid.NetworkConfiguration()
