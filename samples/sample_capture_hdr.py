"""HDR capture sample."""
from zivid import Application, Settings


def _main():
    app = Application()
    with app.connect_camera() as camera:
        settings = Settings(
            acquisitions=[
                Settings.Acquisition(aperture=aperture)
                for aperture in (10.90, 5.80, 2.83)
            ]
        )
        with camera.capture(settings) as hdr_frame:
            hdr_frame.save("result.zdf")


if __name__ == "__main__":
    _main()
