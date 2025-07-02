"""File camera capture sample."""

from pathlib import Path

from zivid import Application, Settings, Settings2D

file_camera_file_path = Path(__file__).parent.parent / "test" / "test_data" / "FileCameraZivid2M70.zfc"


def _main():
    app = Application()
    with app.create_file_camera(file_camera_file_path) as camera:
        settings = Settings(
            acquisitions=[Settings.Acquisition()],
            color=Settings2D(acquisitions=[Settings2D.Acquisition()]),
        )

        with camera.capture_2d_3d(settings) as frame:
            frame.save("result.zdf")


if __name__ == "__main__":
    _main()
