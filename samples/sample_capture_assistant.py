"""Capture Assistant sample."""
import datetime
import zivid
from zivid.captureassistant import AmbientLightFrequency, SuggestSettingsParameters


def _main():
    app = zivid.Application()
    camera = app.connect_camera()

    suggest_settings_parameters = SuggestSettingsParameters(datetime.timedelta(milliseconds=1200),
                                                            AmbientLightFrequency.hz50)

    suggested_settings = zivid.captureassistant.suggest_settings(camera, suggest_settings_parameters)

    frame = zivid.hdr.capture(camera, suggested_settings)
    frame.save("result.zdf")


if __name__ == "__main__":
    _main()
