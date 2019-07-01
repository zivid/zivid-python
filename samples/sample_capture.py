"""Capture sample."""
import datetime
import zivid


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    with camera.update_settings() as updater:
        updater.settings.iris = 40
        updater.settings.exposure_time = datetime.timedelta(microseconds=40000)
        updater.settings.filters.reflection.enabled = True

    with camera.capture() as frame:
        frame.save("result.zdf")


if __name__ == "__main__":
    _main()
