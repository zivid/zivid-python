"""Capture sample."""
import datetime
from zivid import Application, Settings


def _main():
    app = Application()
    camera = app.connect_camera()

    settings = Settings(
        acquisitions=[
            Settings.Acquisition(
                aperture=5.6, exposure_time=datetime.timedelta(microseconds=8333),
            ),
        ],
        processing=Settings.Processing(
            filters=Settings.Processing.Filters(
                outlier=Settings.Processing.Filters.Outlier(
                    removal=Settings.Processing.Filters.Outlier.Removal(
                        enabled=True, threshold=5
                    )
                )
            )
        ),
    )

    with camera.capture(settings) as frame:
        frame.save("result.zdf")


if __name__ == "__main__":
    _main()
