    def __init__(
        self,
        acquisitions=_zivid.Settings().Acquisitions().value,
        processing=Processing(),
    ):

        self._acquisitions = _convert_to_acquistions(acquisitions)
        self.processing = processing

    @property
    def acquisitions(self):
        return self._acquisitions

    @acquisitions.setter
    def acquisitions(self, value):
        self._acquisitions = _convert_to_acquistions(value)

    def __eq__(self, other):
        print("equality being called")
        print("testing aquis equal:")
        print(self._acquisitions == other._acquisitions)
        print("testing processing equal:")
        print(self.processing == other.processing)
        if (
            self._acquisitions == other._acquisitions
            and self.processing == other.processing
        ):
            print("returning True")
            return True
        print("returing false")
        return False

    def __str__(self):
        return """Settings:
    acquisitions: {acquisitions}
    processing: {processing}
    """.format(
            acquisitions="\n".join([str(element) for element in self.acquisitions]),
            processing=self.processing,
        )


def _convert_to_acquistions(inputs):
    temp = []  # Settings().Acquisitions()
    for acquisition_element in inputs:
        if isinstance(acquisition_element, Settings.Acquisition):
            temp.append(acquisition_element)
        elif isinstance(acquisition_element, Settings.Acquisition):
            # temp_settings = Settings()
            # temp_settings.acquisitions = [acquisition_element]
            # acuis = to_internal_settings(temp_settings).acquisitions
            # print(acuis.value[0])
            # temp.append(acuis.value[0])
            temp.append(_to_internal_acquisition(acquisition_element))
        else:
            raise TypeError(
                "Unsupported type {type_of_acquisition_element}".format(
                    type_of_acquisition_element=type(acquisition_element)
                )
            )
    print(temp)
    return temp


def _to_internal_acquisition(acquisition):
    internal_acquisition = _zivid.Settings.Acquisition()

    def _to_internal_patterns(patterns):
        internal_patterns = _zivid.Settings.Acquisition.Patterns()

        def _to_internal_sine(sine):
            internal_sine = _zivid.Settings.Acquisition.Patterns.Sine()
            if sine.bidirectional is not None:

                internal_sine.bidirectional = _zivid.Settings.Acquisition.Patterns.Sine.Bidirectional(
                    sine.bidirectional
                )
            else:
                internal_sine.bidirectional = (
                    _zivid.Settings.Acquisition.Patterns.Sine.Bidirectional()
                )
            pass  # no children
            return internal_sine

        internal_patterns.sine = _to_internal_sine(patterns.sine)
        return internal_patterns

    if acquisition.aperture is not None:
        internal_acquisition.aperture = _zivid.Settings.Acquisition.Aperture(
            acquisition.aperture
        )
    else:
        internal_acquisition.aperture = _zivid.Settings.Acquisition.Aperture()
    if acquisition.brightness is not None:

        internal_acquisition.brightness = _zivid.Settings.Acquisition.Brightness(
            acquisition.brightness
        )
    else:
        internal_acquisition.brightness = _zivid.Settings.Acquisition.Brightness()
    if acquisition.exposure_time is not None:

        internal_acquisition.exposure_time = _zivid.Settings.Acquisition.ExposureTime(
            acquisition.exposure_time
        )
    else:
        internal_acquisition.exposure_time = _zivid.Settings.Acquisition.ExposureTime()
    if acquisition.gain is not None:

        internal_acquisition.gain = _zivid.Settings.Acquisition.Gain(acquisition.gain)
    else:
        internal_acquisition.gain = _zivid.Settings.Acquisition.Gain()

    internal_acquisition.patterns = _to_internal_patterns(acquisition.patterns)
    return internal_acquisition






















type_map_basic_types = 
{
    int: (int, float),
    ...
}

type_map = type_map_basic_types + {typing.Optional[k], (typing.Optional[vV] for vv in v) for k,v in type_map_basic_types}


def gen():



class Settings:
    class Acquisition:
        def __init__(
            self,
            aperture=_zivid.Settings().Acquisition().Aperture().value_type [int, None, float], # " "
            brightness : =_zivid.Settings().Acquisition().Brightness().value,
            exposure_time=_zivid.Settings().Acquisition().ExposureTime().value,
            gain=_zivid.Settings().Acquisition().Gain().value,
        ):
            if isinstance(aperture, *type_map(_zivid.Settings.Acquisition.Aperture.value_type))
                self._aperture = _zivid.Settings.Acquisition.Aperture(aperture)
            else:
                raise "ERR"

            try:
                self._aperture = _zivid.Settings.Acquisition.Aperture(aperture)
            except TypeError:
                raise TypeError(get_type_error_message(_zivid.Settings.Acquisition.Aperture.value_type))

            self._brightness = _zivid.Settings.Acquisition.Brightness(brightness)
            self._exposure_time = _zivid.Settings.Acquisition.ExposureTime(
                exposure_time
            )
            self._gain = _zivid.Settings.Acquisition.Gain(gain)