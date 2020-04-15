"""Capture sample."""
import datetime
from zivid import Application, Settings


def _main():
    app = Application()
    camera = app.connect_camera()

    settings = Settings()
    settings.acquisitions.append(Settings.Acquisition())
    settings.acquisitions[0].aperture = 5.6
    settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=8333)
    settings.processing.filters.outlier.removal.enabled = True
    settings.processing.filters.outlier.removal.threshold = 5.0

    with camera.capture(settings) as frame:
        frame.save("result.zdf")


if __name__ == "__main__":
    _main()
