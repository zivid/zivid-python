def test_default_init_settings(application):  # pylint: disable=unused-argument
    import numbers
    import datetime
    import zivid

    settings = zivid.Settings()
    assert settings.bidirectional is not None
    assert settings.blue_balance is not None
    assert settings.brightness is not None
    assert settings.exposure_time is not None
    assert settings.filters
    assert settings.gain is not None
    assert settings.iris is not None
    assert settings.red_balance is not None
    assert isinstance(settings.bidirectional, bool)
    assert isinstance(settings.blue_balance, numbers.Real)
    assert isinstance(settings.brightness, numbers.Real)
    assert isinstance(settings.exposure_time, datetime.timedelta)
    assert isinstance(settings.filters, zivid.Settings.Filters)
    assert isinstance(settings.gain, numbers.Real)
    assert isinstance(settings.iris, numbers.Real)
    assert isinstance(settings.red_balance, numbers.Real)


def test_init_bidirectional(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = True
    settings = zivid.Settings(bidirectional=value)
    assert settings.bidirectional == value
    assert isinstance(settings.bidirectional, numbers.Real)


def test_init_blue_balance(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 33
    settings = zivid.Settings(blue_balance=value)
    assert settings.blue_balance == value
    assert isinstance(settings.blue_balance, numbers.Real)


def test_init_brightness(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 34
    settings = zivid.Settings(brightness=value)
    assert settings.brightness == value
    assert isinstance(settings.brightness, numbers.Real)


def test_init_exposure_time(application):  # pylint: disable=unused-argument
    import datetime
    import zivid

    value = datetime.timedelta(microseconds=10000)
    settings = zivid.Settings(exposure_time=value)
    assert settings.exposure_time == value
    assert isinstance(settings.exposure_time, datetime.timedelta)


def test_init_filters(application):  # pylint: disable=unused-argument
    import zivid

    value = zivid.Settings.Filters()
    settings = zivid.Settings(filters=value)
    assert settings.filters == value
    assert isinstance(settings.filters, zivid.Settings.Filters)


def test_init_gain(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 36
    settings = zivid.Settings(gain=value)
    assert settings.gain == value
    assert isinstance(settings.gain, numbers.Real)


def test_init_iris(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 37
    settings = zivid.Settings(iris=value)
    assert settings.iris == value
    assert isinstance(settings.iris, numbers.Real)


def test_init_red_balance(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    value = 38
    settings = zivid.Settings(red_balance=value)
    assert settings.red_balance == value
    assert isinstance(settings.red_balance, numbers.Real)


def test_not_equal_bidirectional(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(bidirectional=True)
    settings2 = zivid.Settings(bidirectional=False)

    assert settings1 != settings2


def test_equal_bidirectional(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(bidirectional=True)
    settings2 = zivid.Settings(bidirectional=True)
    assert settings1 == settings2


def test_not_equal_blue_balance(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(blue_balance=34)
    settings2 = zivid.Settings(blue_balance=43)

    assert settings1 != settings2


def test_equal_blue_balance(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(blue_balance=34)
    settings2 = zivid.Settings(blue_balance=34)
    assert settings1 == settings2


def test_not_equal_brightness(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(brightness=34)
    settings2 = zivid.Settings(brightness=43)

    assert settings1 != settings2


def test_equal_brightness(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(brightness=34)
    settings2 = zivid.Settings(brightness=34)
    assert settings1 == settings2


def test_not_equal_exposure_time(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(exposure_time=3333)
    settings2 = zivid.Settings(exposure_time=9999)

    assert settings1 != settings2


def test_equal_exposure_time(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(exposure_time=3333)
    settings2 = zivid.Settings(exposure_time=3333)
    assert settings1 == settings2


def test_not_equal_filters(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(
        filters=zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=True))
    )
    settings2 = zivid.Settings(
        filters=zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=False))
    )

    assert settings1 != settings2


def test_equal_filters(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(
        filters=zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=False))
    )
    settings2 = zivid.Settings(
        filters=zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=False))
    )
    assert settings1 == settings2


def test_not_equal_gain(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(gain=0)
    settings2 = zivid.Settings(gain=1)

    assert settings1 != settings2


def test_equal_gain(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(gain=1)
    settings2 = zivid.Settings(gain=1)
    assert settings1 == settings2


def test_not_equal_iris(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(iris=34)
    settings2 = zivid.Settings(iris=43)

    assert settings1 != settings2


def test_equal_iris(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(iris=34)
    settings2 = zivid.Settings(iris=34)
    assert settings1 == settings2


def test_not_equal_red_balance(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(red_balance=34)
    settings2 = zivid.Settings(red_balance=43)

    assert settings1 != settings2


def test_equal_red_balance(application):  # pylint: disable=unused-argument
    import zivid

    settings1 = zivid.Settings(red_balance=34)
    settings2 = zivid.Settings(red_balance=34)
    assert settings1 == settings2
