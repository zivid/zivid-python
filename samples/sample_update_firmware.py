"""Update firmware on the Zivid camera."""

import zivid


def _main() -> None:
    app = zivid.Application()

    cameras = app.cameras()
    if len(cameras) == 0:
        raise TimeoutError("No camera found.")

    print(f"Found {len(cameras)} camera(s)")
    for camera in cameras:
        if not zivid.firmware.is_up_to_date(camera):
            print("Firmware update required")
            print(
                f"Updating firmware on camera {camera.info.serial_number}, model name: {camera.info.model_name},"
                f" firmware version: {camera.info.firmware_version}"
            )
            zivid.firmware.update(
                camera,
                progress_callback=lambda progress, description: print(
                    f'{progress}% : {description}{("", "...")[progress < 100]}'
                ),
            )
        else:
            print(
                f"Skipping update of camera {camera.info.serial_number}, model name: {camera.info.model_name},"
                f" firmware version: {camera.info.firmware_version}"
            )


if __name__ == "__main__":
    _main()
