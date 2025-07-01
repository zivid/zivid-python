"""Capture Assistant sample."""

import datetime

import zivid
from zivid.capture_assistant import SuggestSettingsParameters


def _main():
    app = zivid.Application()
    with app.connect_camera() as camera:
        suggest_settings_parameters = SuggestSettingsParameters(
            max_capture_time=datetime.timedelta(milliseconds=1200),
            ambient_light_frequency=SuggestSettingsParameters.AmbientLightFrequency.none,
        )

        settings = zivid.capture_assistant.suggest_settings(camera, suggest_settings_parameters)

        with camera.capture_2d_3d(settings) as frame:
            frame.save("result.zdf")


if __name__ == "__main__":
    _main()
