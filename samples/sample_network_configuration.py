"""Sample demonstrating how to configure network settings of a Zivid camera."""
import zivid


def _confirm(message):
    while True:
        input_value = input(f"{message} [Y/n] ")
        if input_value.lower() in ["y", "yes"]:
            return True
        if input_value.lower() in ["n", "no"]:
            return False


def _main():
    try:
        app = zivid.Application()
        camera = app.cameras()[0]

        original_config = camera.network_configuration

        print(f"Current network configuration of camera {camera.info.serial_number}:")
        print(original_config)
        print()

        mode = zivid.NetworkConfiguration.IPV4.Mode.manual
        address = original_config.ipv4.address
        subnet_mask = original_config.ipv4.subnet_mask

        if _confirm("Do you want to use DHCP?"):
            mode = zivid.NetworkConfiguration.IPV4.Mode.dhcp
        else:
            input_address = input(
                f"Enter IPv4 Address [{original_config.ipv4.address}]: "
            )
            address = input_address if input_address else original_config.ipv4.address
            input_subnet_mask = input(
                f"Enter new Subnet mask [{original_config.ipv4.subnet_mask}]: "
            )
            subnet_mask = (
                input_subnet_mask
                if input_subnet_mask
                else original_config.ipv4.subnet_mask
            )

        new_config = zivid.NetworkConfiguration(
            ipv4=zivid.NetworkConfiguration.IPV4(
                mode=mode,
                address=address,
                subnet_mask=subnet_mask,
            )
        )

        print()
        print("New network configuration:")
        print(new_config)
        if not _confirm(
            f"Do you want to apply the new network configuration to camera {camera.info.serial_number}?"
        ):
            return 0

        print("Applying network configuration...")
        camera.apply_network_configuration(new_config)

        print(f"Updated network configuration of camera {camera.info.serial_number}:")
        print(camera.network_configuration)
        print()

        print(f"Camera status is '{camera.state.status}'")
    except RuntimeError as ex:
        print(f"Error: {str(ex)}")
        return 1
    return 0


if __name__ == "__main__":
    _main()
