"""HDR capture sample."""
import zivid


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    settings_collection = [camera.settings for _ in range(3)]
    settings_collection[0].iris = 14
    settings_collection[1].iris = 21
    settings_collection[2].iris = 35

    with camera.capture(settings_collection) as hdr_frame:
        hdr_frame.save("result.zdf")


if __name__ == "__main__":
    _main()
