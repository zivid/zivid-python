"""File camera capture sample."""
import zivid


def _main():
    app = zivid.Application()
    file_camera = app.create_file_camera("MiscObjects.zdf")

    with file_camera.capture() as frame:
        frame.save("results.zdf")


if __name__ == "__main__":
    _main()
