"""File camera capture sample."""
from zivid import Application, Settings


def _main():
    app = Application()
    with app.create_file_camera("FileCameraZivid2M70.zfc") as camera:
        settings = Settings(acquisitions=[Settings.Acquisition()])

        with camera.capture(settings) as frame:
            frame.save("result.zdf")


if __name__ == "__main__":
    _main()
