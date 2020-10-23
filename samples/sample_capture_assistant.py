"""Capture Assistant sample."""
import datetime
import zivid
from zivid.capture_assistant import SuggestSettingsParameters


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    suggest_settings_parameters = SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=1200),
        ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.none,
    )

    settings = zivid.capture_assistant.suggest_settings(
        camera, suggest_settings_parameters
    )

    with camera.capture(settings) as frame:
        frame.save("result.zdf")


if __name__ == "__main__":
    _main()
