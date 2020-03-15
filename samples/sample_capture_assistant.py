"""Capture Assistant sample."""
import datetime
import zivid
from zivid.capture_assistant import AmbientLightFrequency, SuggestSettingsParameters


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=AmbientLightFrequency.none,
    )

    settings_list = zivid.capture_assistant.suggest_settings(
        camera, suggest_settings_parameters
    )

    with zivid.hdr.capture(camera, settings_list) as hdr_frame:
        hdr_frame.save("result.zdf")


if __name__ == "__main__":
    _main()
