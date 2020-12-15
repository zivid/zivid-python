import pytest


def test_init_default_settings(application):
    import zivid

    settings_2d = zivid.Settings2D()

    assert isinstance(settings_2d.acquisitions, list)
    assert isinstance(settings_2d.processing, zivid.Settings2D.Processing)

    assert isinstance(settings_2d.processing.color, zivid.Settings2D.Processing.Color)
    assert isinstance(
        settings_2d.processing.color.balance, zivid.Settings2D.Processing.Color.Balance
    )
    assert settings_2d.processing.color.gamma is None
    assert settings_2d.processing.color.balance.red is None
    assert settings_2d.processing.color.balance.green is None
    assert settings_2d.processing.color.balance.blue is None


def test_default_acquisition(application):
    import zivid
    import datetime

    settings_2d = zivid.Settings2D(acquisitions=[zivid.Settings2D.Acquisition()])
    assert isinstance(settings_2d.acquisitions, list)
    acquisition = settings_2d.acquisitions[0]

    assert isinstance(acquisition, zivid.Settings2D.Acquisition)
    assert acquisition.aperture is None
    assert acquisition.brightness is None
    assert acquisition.exposure_time is None
    assert acquisition.gain is None

    pytest.helpers.equality_tester(
        zivid.Settings2D.Acquisition,
        [5, 0.5, datetime.timedelta(microseconds=11000), 14],
        [5, 0.5, datetime.timedelta(microseconds=11001), 14],
    )


def test_acquisition_brightness(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Acquisition(),
        member="brightness",
        value=0.5,
        expected_data_type=numbers.Real,
    )


def test_acquisition_exposure_time(application):
    import datetime
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Acquisition(),
        member="exposure_time",
        value=datetime.timedelta(microseconds=100000),
        expected_data_type=datetime.timedelta,
    )


def test_acquisition_gain(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Acquisition(),
        member="gain",
        value=14,
        expected_data_type=numbers.Real,
    )


def test_acquisition_aperture(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Acquisition(),
        member="aperture",
        value=20.5,
        expected_data_type=numbers.Real,
    )


def test_settings_processing(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D(),
        member="processing",
        value=zivid.Settings2D.Processing(),
        expected_data_type=zivid.Settings2D.Processing,
    )
    pytest.helpers.equality_tester(
        zivid.Settings2D.Processing,
        [
            zivid.Settings2D.Processing.Color(
                0.9, zivid.Settings2D.Processing.Color.Balance(blue=1.1)
            )
        ],
        [
            zivid.Settings2D.Processing.Color(
                1.1, zivid.Settings2D.Processing.Color.Balance(blue=1.2)
            )
        ],
    )


def test_settings_processing_color(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing(),
        member="color",
        value=zivid.Settings2D.Processing.Color(),
        expected_data_type=zivid.Settings2D.Processing.Color,
    )
    pytest.helpers.equality_tester(
        zivid.Settings2D.Processing.Color,
        [0.9, zivid.Settings2D.Processing.Color.Balance(red=1.1)],
        [1.1, zivid.Settings2D.Processing.Color.Balance(red=1.2)],
    )


def test_settings_processing_color_gamma(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing.Color(),
        member="gamma",
        value=0.85,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing.Color(),
        member="balance",
        value=zivid.Settings2D.Processing.Color.Balance(),
        expected_data_type=zivid.Settings2D.Processing.Color.Balance,
    )
    pytest.helpers.equality_tester(
        zivid.Settings2D.Processing.Color.Balance, [1.1, 1.1, 1.1], [1.2, 1.1, 1.1],
    )


def test_settings_processing_color_balance_red(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing.Color.Balance(),
        member="red",
        value=2,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_green(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing.Color.Balance(),
        member="green",
        value=1.5,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_blue(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings2D.Processing.Color.Balance(),
        member="blue",
        value=1,
        expected_data_type=numbers.Real,
    )
