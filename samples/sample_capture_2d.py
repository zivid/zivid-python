"""Capture sample 2D."""
import datetime
import zivid


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    settings_2d = zivid.Settings2D()
    settings_2d.iris = 50
    settings_2d.exposure_time = datetime.timedelta(microseconds=50000)

    with camera.capture_2d(settings_2d) as frame_2d:
        image = frame_2d.image()
        image.save("result.png")


if __name__ == "__main__":
    _main()
