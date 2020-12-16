"""Auto generated, do not edit."""
import zivid
import _zivid


def to_settings2_d_acquisition(internal_acquisition):
    return zivid.Settings2D.Acquisition(
        aperture=internal_acquisition.aperture.value,
        brightness=internal_acquisition.brightness.value,
        exposure_time=internal_acquisition.exposure_time.value,
        gain=internal_acquisition.gain.value,
    )


def to_settings2_d_processing_color_balance(internal_balance):
    return zivid.Settings2D.Processing.Color.Balance(
        blue=internal_balance.blue.value,
        green=internal_balance.green.value,
        red=internal_balance.red.value,
    )


def to_settings2_d_processing_color(internal_color):
    return zivid.Settings2D.Processing.Color(
        balance=to_settings2_d_processing_color_balance(internal_color.balance),
        gamma=internal_color.gamma.value,
    )


def to_settings2_d_processing(internal_processing):
    return zivid.Settings2D.Processing(
        color=to_settings2_d_processing_color(internal_processing.color),
    )


def to_settings2_d(internal_settings2_d):
    return zivid.Settings2D(
        processing=to_settings2_d_processing(internal_settings2_d.processing),
        acquisitions=[
            to_settings2_d_acquisition(element)
            for element in internal_settings2_d.acquisitions.value
        ],
    )


def to_internal_settings2_d_acquisition(acquisition):
    internal_acquisition = _zivid.Settings2D.Acquisition()

    internal_acquisition.aperture = _zivid.Settings2D.Acquisition.Aperture(
        acquisition.aperture
    )
    internal_acquisition.brightness = _zivid.Settings2D.Acquisition.Brightness(
        acquisition.brightness
    )
    internal_acquisition.exposure_time = _zivid.Settings2D.Acquisition.ExposureTime(
        acquisition.exposure_time
    )
    internal_acquisition.gain = _zivid.Settings2D.Acquisition.Gain(acquisition.gain)

    return internal_acquisition


def to_internal_settings2_d_processing_color_balance(balance):
    internal_balance = _zivid.Settings2D.Processing.Color.Balance()

    internal_balance.blue = _zivid.Settings2D.Processing.Color.Balance.Blue(
        balance.blue
    )
    internal_balance.green = _zivid.Settings2D.Processing.Color.Balance.Green(
        balance.green
    )
    internal_balance.red = _zivid.Settings2D.Processing.Color.Balance.Red(balance.red)

    return internal_balance


def to_internal_settings2_d_processing_color(color):
    internal_color = _zivid.Settings2D.Processing.Color()

    internal_color.gamma = _zivid.Settings2D.Processing.Color.Gamma(color.gamma)
    internal_color.balance = to_internal_settings2_d_processing_color_balance(
        color.balance
    )
    return internal_color


def to_internal_settings2_d_processing(processing):
    internal_processing = _zivid.Settings2D.Processing()

    internal_processing.color = to_internal_settings2_d_processing_color(
        processing.color
    )
    return internal_processing


def to_internal_settings2_d(settings2_d):
    internal_settings2_d = _zivid.Settings2D()
    internal_settings2_d.processing = to_internal_settings2_d_processing(
        settings2_d.processing
    )
    temp = _zivid.Settings2D().Acquisitions()
    for acq in settings2_d.acquisitions:
        temp.append(to_internal_settings2_d_acquisition(acq))
    internal_settings2_d.acquisitions = temp

    return internal_settings2_d
