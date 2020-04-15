"""Capture sample 2D."""
import datetime
from zivid import Application, Settings2D


def _main():
    app = Application()
    camera = app.connect_camera()

    settings_2d = Settings2D(
        acquisitions=Settings2D.Acquisition(
            aperture=2.83, exposure_time=datetime.timedelta(microseconds=10000),
        )
    )

    with camera.capture(settings_2d) as frame_2d:
        image = frame_2d.image_rgba()
        image.save("result.png")


if __name__ == "__main__":
    _main()
