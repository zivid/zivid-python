"""Contains functions to convert between settings 2d and internal settings 2d."""
import _zivid
from zivid.settings_2d import Settings2D


import _zivid


def to_internal_settings_2d(settings2_d):
    internal_settings2_d = _zivid.Settings2D()

    def _to_internal_acquisition(acquisition):
        internal_acquisition = _zivid.Settings2D.Acquisition()

        if acquisition.aperture is not None:

            internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture(
                acquisition.aperture
            )
        else:
            internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture()
        if acquisition.brightness is not None:

            internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness(
                acquisition.brightness
            )
        else:
            internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness()
        if acquisition.exposure_time is not None:

            internal_acquisition.exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
                acquisition.exposure_time
            )
        else:
            internal_acquisition.exposure_time = (
                _zivid.Settings2D.Acquisition.ExposureTime()
            )
        if acquisition.gain is not None:

            internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain(
                acquisition.gain
            )
        else:
            internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain()

        return internal_acquisition

    def _to_internal_processing(processing):
        internal_processing = _zivid.Settings2D.Processing()

        def _to_internal_color(color):
            internal_color = _zivid.Settings2D.Processing.Color()

            def _to_internal_balance(balance):
                internal_balance = _zivid.Settings2D.Processing.Color.Balance()

                if balance.blue is not None:

                    internal_balance.blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
                        balance.blue
                    )
                else:
                    internal_balance.blue = (
                        _zivid.Settings2D.Processing.Color.Balance.Blue()
                    )
                if balance.green is not None:

                    internal_balance.green = _zivid.Settings2D.Processing.Color.Balance.Green(
                        balance.green
                    )
                else:
                    internal_balance.green = (
                        _zivid.Settings2D.Processing.Color.Balance.Green()
                    )
                if balance.red is not None:

                    internal_balance.red = _zivid.Settings2D.Processing.Color.Balance.Red(
                        balance.red
                    )
                else:
                    internal_balance.red = (
                        _zivid.Settings2D.Processing.Color.Balance.Red()
                    )

                return internal_balance

            global to_internal_processing_color_balance
            to_internal_processing_color_balance = _to_internal_balance

            internal_color.balance = _to_internal_balance(color.balance)
            return internal_color

        global to_internal_processing_color
        to_internal_processing_color = _to_internal_color

        internal_processing.color = _to_internal_color(processing.color)
        return internal_processing

    global to_internal_acquisition
    to_internal_acquisition = _to_internal_acquisition
    global to_internal_processing
    to_internal_processing = _to_internal_processing

    internal_settings2_d.processing = _to_internal_processing(settings2_d.processing)

    if settings2_d.acquisitions is None:
        internal_settings2_d.acquisitions = _zivid.Settings().Acquisitions()
    else:
        temp = _zivid.Settings2D().Acquisitions()
        for acq in settings2_d.acquisitions:
            temp.append(_to_internal_acquisition(acq))
        internal_settings2_d.acquisitions = temp

    return internal_settings2_d


import zivid


def to_settings_2d(internal_settings2_d):
    def _to_acquisition(internal_acquisition):

        return zivid.Settings2D.Acquisition(
            aperture=internal_acquisition.aperture.value,
            brightness=internal_acquisition.brightness.value,
            exposure_time=internal_acquisition.exposure_time.value,
            gain=internal_acquisition.gain.value,
        )

    def _to_processing(internal_processing):
        def _to_color(internal_color):
            def _to_balance(internal_balance):

                return zivid.Settings2D.Processing.Color.Balance(
                    blue=internal_balance.blue.value,
                    green=internal_balance.green.value,
                    red=internal_balance.red.value,
                )

            global to_processing_color_balance
            to_processing_color_balance = _to_balance
            return zivid.Settings2D.Processing.Color(
                balance=_to_balance(internal_color.balance),
            )

        global to_processing_color
        to_processing_color = _to_color
        return zivid.Settings2D.Processing(color=_to_color(internal_processing.color),)

    global to_acquisition
    to_acquisition = _to_acquisition
    global to_processing
    to_processing = _to_processing

    return zivid.Settings2D(
        processing=_to_processing(internal_settings2_d.processing),
        acquisitions=[
            _to_acquisition(element)
            for element in internal_settings2_d.acquisitions.value
        ],
    )
