"""Sample demonstrating how to capture with multiple cameras at the same time."""

import concurrent.futures
import datetime
import time

import zivid


def _capture_sync(cameras: list[zivid.Camera]) -> list[zivid.Frame]:
    return [
        camera.capture(
            zivid.Settings(
                acquisitions=[
                    zivid.Settings.Acquisition(
                        exposure_time=datetime.timedelta(microseconds=100000)
                    )
                ]
            )
        )
        for camera in cameras
    ]


def _capture_async(cameras: list[zivid.Camera]) -> list[zivid.Frame]:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                camera.capture,
                zivid.Settings(
                    acquisitions=[
                        zivid.Settings.Acquisition(
                            exposure_time=datetime.timedelta(microseconds=100000)
                        )
                    ]
                ),
            )
            for camera in cameras
        ]

        return [future.result() for future in futures]


def _main():
    app = zivid.Application()
    cameras = app.cameras()
    for camera in cameras:
        camera.connect()

    start = time.monotonic()
    _capture_async(cameras)
    end = time.monotonic()
    print(
        f"Time taken to capture asynchronously from {len(cameras)} camera(s): {end - start} seconds"
    )

    start = time.monotonic()
    _capture_sync(cameras)
    end = time.monotonic()
    print(
        f"Time taken to capture synchronously from {len(cameras)} camera(s): {end - start} seconds"
    )

    for camera in cameras:
        camera.disconnect()


if __name__ == "__main__":
    _main()
