"""HDR capture sample."""

import datetime

from zivid import Application, Settings, Settings2D


def _main():
    app = Application()
    with app.connect_camera() as camera:
        settings = Settings(
            acquisitions=[
                Settings.Acquisition(aperture=aperture)
                for aperture in (10.90, 5.80, 2.83)
            ],
            color=Settings2D(
                acquisitions=[
                    Settings2D.Acquisition(
                        exposure_time=datetime.timedelta(microseconds=exposure_time)
                    )
                    for exposure_time in (1677, 5000, 10000)
                ]
            ),
        )
        with camera.capture(settings) as hdr_frame:
            hdr_frame.save("result.zdf")


if __name__ == "__main__":
    _main()
