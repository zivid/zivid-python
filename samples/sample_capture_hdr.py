"""HDR capture sample."""
import zivid


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    settings_list = [camera.settings for _ in range(3)]
    settings_list[0].iris = 14
    settings_list[1].iris = 21
    settings_list[2].iris = 35

    with camera.capture(settings_list) as hdr_frame:
        hdr_frame.save("result.zdf")


if __name__ == "__main__":
    _main()
