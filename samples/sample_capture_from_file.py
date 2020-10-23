"""File camera capture sample."""
from zivid import Application, Settings


def _main():
    app = Application()
    camera = app.create_file_camera("FileCameraZividOne.zfc")

    settings = Settings(acquisitions=[Settings.Acquisition()])

    with camera.capture(settings) as frame:
        frame.save("result.zdf")


if __name__ == "__main__":
    _main()
