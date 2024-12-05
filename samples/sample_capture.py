"""Capture sample."""

import datetime
from zivid import Application, Settings, Settings2D


def _main():
    app = Application()
    with app.connect_camera() as camera:
        settings = Settings()
        settings.acquisitions.append(Settings.Acquisition())
        settings.acquisitions[0].aperture = 5.6
        settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=8333)
        settings.processing.filters.outlier.removal.enabled = True
        settings.processing.filters.outlier.removal.threshold = 5.0

        settings.color = Settings2D()
        settings.color.acquisitions.append(Settings2D.Acquisition())
        settings.color.acquisitions[0].aperture = 5.6
        settings.color.acquisitions[0].exposure_time = datetime.timedelta(
            microseconds=8333
        )

        with camera.capture_2d_3d(settings) as frame:
            frame.save("result.zdf")


if __name__ == "__main__":
    _main()
