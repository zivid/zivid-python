# pylint: disable=import-outside-toplevel


def test_default_init_settings_2d(application):  # pylint: disable=unused-argument
    import numbers
    import datetime
    import zivid

    settings_2d = zivid.Settings2D()
    assert settings_2d.brightness is not None
    assert settings_2d.exposure_time is not None
    assert settings_2d.gain is not None
    assert settings_2d.iris is not None
    assert isinstance(settings_2d.brightness, numbers.Real)
    assert isinstance(settings_2d.exposure_time, datetime.timedelta)
    assert isinstance(settings_2d.gain, numbers.Real)
    assert isinstance(settings_2d.iris, numbers.Real)


def test_init_brightness(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 34
    settings_2d = zivid.Settings2D(brightness=value)
    assert settings_2d.brightness == value
    assert isinstance(settings_2d.brightness, numbers.Real)


def test_init_exposure_time(application):  # pylint: disable=unused-argument
    import datetime
    import zivid

    value = datetime.timedelta(microseconds=10000)
    settings_2d = zivid.Settings2D(exposure_time=value)
    assert settings_2d.exposure_time == value
    assert isinstance(settings_2d.exposure_time, datetime.timedelta)


def test_init_gain(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 36
    settings_2d = zivid.Settings2D(gain=value)
    assert settings_2d.gain == value
    assert isinstance(settings_2d.gain, numbers.Real)


def test_init_iris(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 37
    settings_2d = zivid.Settings2D(iris=value)
    assert settings_2d.iris == value
    assert isinstance(settings_2d.iris, numbers.Real)


def test_not_equal_brightness(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(brightness=34)
    settings_2d_2 = zivid.Settings2D(brightness=43)

    assert settings_2d_1 != settings_2d_2


def test_equal_brightness(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(brightness=34)
    settings_2d_2 = zivid.Settings2D(brightness=34)
    assert settings_2d_1 == settings_2d_2


def test_not_equal_exposure_time(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(exposure_time=3333)
    settings_2d_2 = zivid.Settings2D(exposure_time=9999)

    assert settings_2d_1 != settings_2d_2


def test_equal_exposure_time(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(exposure_time=3333)
    settings_2d_2 = zivid.Settings2D(exposure_time=3333)
    assert settings_2d_1 == settings_2d_2


def test_not_equal_gain(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(gain=0)
    settings_2d_2 = zivid.Settings2D(gain=1)

    assert settings_2d_1 != settings_2d_2


def test_equal_gain(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(gain=1)
    settings_2d_2 = zivid.Settings2D(gain=1)
    assert settings_2d_1 == settings_2d_2


def test_not_equal_iris(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(iris=34)
    settings_2d_2 = zivid.Settings2D(iris=43)

    assert settings_2d_1 != settings_2d_2


def test_equal_iris(application):  # pylint: disable=unused-argument
    import zivid

    settings_2d_1 = zivid.Settings2D(iris=34)
    settings_2d_2 = zivid.Settings2D(iris=34)
    assert settings_2d_1 == settings_2d_2
