"""File camera capture sample."""

from zivid import Application, Settings, Settings2D


def _main():
    app = Application()
    with app.create_file_camera("FileCameraZivid2M70.zfc") as camera:
        settings = Settings(
            acquisitions=[Settings.Acquisition()],
            color=Settings2D(acquisitions=[Settings2D.Acquisition()]),
        )

        with camera.capture_2d_3d(settings) as frame:
            frame.save("result.zdf")


if __name__ == "__main__":
    _main()
