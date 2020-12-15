import pytest


def test_default_settings(application):
    import zivid

    settings = zivid.Settings()

    assert isinstance(settings.acquisitions, list)
    assert isinstance(settings.processing, zivid.Settings.Processing)
    assert isinstance(settings.processing.color, zivid.Settings.Processing.Color)
    assert isinstance(
        settings.processing.color.balance, zivid.Settings.Processing.Color.Balance
    )
    assert settings.processing.color.gamma is None
    assert settings.processing.color.balance.red is None
    assert settings.processing.color.balance.green is None
    assert settings.processing.color.balance.blue is None

    assert isinstance(settings.processing.filters, zivid.Settings.Processing.Filters)
    assert isinstance(
        settings.processing.filters.experimental,
        zivid.Settings.Processing.Filters.Experimental,
    )
    assert isinstance(
        settings.processing.filters.experimental.contrast_distortion,
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion,
    )
    assert isinstance(
        settings.processing.filters.experimental.contrast_distortion.correction,
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction,
    )
    assert (
        settings.processing.filters.experimental.contrast_distortion.correction.enabled
        is None
    )

    assert isinstance(
        settings.processing.filters.experimental.contrast_distortion.removal,
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal,
    )
    assert (
        settings.processing.filters.experimental.contrast_distortion.removal.enabled
        is None
    )

    assert isinstance(
        settings.processing.filters.noise, zivid.Settings.Processing.Filters.Noise
    )
    assert isinstance(
        settings.processing.filters.noise.removal,
        zivid.Settings.Processing.Filters.Noise.Removal,
    )
    assert settings.processing.filters.noise.removal.enabled is None

    assert isinstance(
        settings.processing.filters.outlier, zivid.Settings.Processing.Filters.Outlier
    )
    assert isinstance(
        settings.processing.filters.outlier.removal,
        zivid.Settings.Processing.Filters.Outlier.Removal,
    )
    assert settings.processing.filters.outlier.removal.enabled is None

    assert isinstance(
        settings.processing.filters.reflection,
        zivid.Settings.Processing.Filters.Reflection,
    )
    assert isinstance(
        settings.processing.filters.reflection.removal,
        zivid.Settings.Processing.Filters.Reflection.Removal,
    )
    assert settings.processing.filters.reflection.removal.enabled is None

    assert isinstance(
        settings.processing.filters.smoothing,
        zivid.Settings.Processing.Filters.Smoothing,
    )
    assert isinstance(
        settings.processing.filters.smoothing.gaussian,
        zivid.Settings.Processing.Filters.Smoothing.Gaussian,
    )
    assert settings.processing.filters.smoothing.gaussian.enabled is None
    assert settings.processing.filters.smoothing.gaussian.sigma is None


def test_set_acquisition_list():
    from zivid import Settings

    settings = Settings()

    settings.acquisitions = [Settings.Acquisition(), settings.Acquisition()]
    assert settings.acquisitions is not None
    assert isinstance(settings.acquisitions, list)
    for element in settings.acquisitions:
        assert isinstance(element, Settings.Acquisition)


def test_set_acquisition_generator():
    from zivid import Settings

    settings = Settings()

    def _generator():
        for _ in range(3):
            yield Settings.Acquisition()

    settings.acquisitions = _generator()
    assert settings.acquisitions is not None
    assert isinstance(settings.acquisitions, list)
    for element in settings.acquisitions:
        assert isinstance(element, Settings.Acquisition)


def test_set_acquisition_tuple():
    from zivid import Settings

    settings = Settings()

    settings.acquisitions = (Settings.Acquisition(), settings.Acquisition())
    assert settings.acquisitions is not None
    assert isinstance(settings.acquisitions, list)
    for element in settings.acquisitions:
        assert isinstance(element, Settings.Acquisition)


def test_default_acquisition(application):
    import zivid
    import datetime

    settings = zivid.Settings(acquisitions=[zivid.Settings.Acquisition()])
    assert isinstance(settings.acquisitions, list)
    acquisition = settings.acquisitions[0]

    assert isinstance(acquisition, zivid.Settings.Acquisition)
    assert acquisition.aperture is None
    assert acquisition.brightness is None
    assert acquisition.gain is None
    assert acquisition.exposure_time is None
    pytest.helpers.equality_tester(
        zivid.Settings.Acquisition,
        [5, 0.5, datetime.timedelta(microseconds=11000), 15],
        [5, 0.5, datetime.timedelta(microseconds=11001), 15],
    )


def test_acquisition_brightness(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Acquisition(),
        member="brightness",
        value=0.5,
        expected_data_type=numbers.Real,
    )


def test_acquisition_exposure_time(application):
    import datetime
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Acquisition(),
        member="exposure_time",
        value=datetime.timedelta(microseconds=100000),
        expected_data_type=datetime.timedelta,
    )


def test_acquisition_gain(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Acquisition(),
        member="gain",
        value=14,
        expected_data_type=numbers.Real,
    )


def test_acquisition_aperture(application):
    import numbers
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Acquisition(),
        member="aperture",
        value=20.5,
        expected_data_type=numbers.Real,
    )


def test_settings_processing(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings(),
        member="processing",
        value=zivid.Settings.Processing(),
        expected_data_type=zivid.Settings.Processing,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing,
        [
            zivid.Settings.Processing.Color(
                0.9, zivid.Settings.Processing.Color.Balance(blue=1.1)
            )
        ],
        [
            zivid.Settings.Processing.Color(
                1.1, zivid.Settings.Processing.Color.Balance(blue=1.2)
            )
        ],
    )


def test_settings_processing_color(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing(),
        member="color",
        value=zivid.Settings.Processing.Color(),
        expected_data_type=zivid.Settings.Processing.Color,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color,
        [0.9, zivid.Settings.Processing.Color.Balance(blue=1.1)],
        [1.1, zivid.Settings.Processing.Color.Balance(blue=1.2)],
    )


def test_settings_processing_color_gamma(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color(),
        member="gamma",
        value=0.85,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color(),
        member="balance",
        value=zivid.Settings.Processing.Color.Balance(),
        expected_data_type=zivid.Settings.Processing.Color.Balance,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color.Balance, [1.1, 1.1, 1.1], [1.2, 1.1, 1.1],
    )


def test_settings_processing_color_balance_red(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="red",
        value=2,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_green(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="green",
        value=1.5,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_blue(application,):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="blue",
        value=1,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_filters(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing(),
        member="filters",
        value=zivid.Settings.Processing.Filters(),
        expected_data_type=zivid.Settings.Processing.Filters,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters,
        [
            zivid.Settings.Processing.Filters.Experimental(
                contrast_distortion=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(
                    correction=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
                        enabled=True
                    )
                )
            )
        ],
        [
            zivid.Settings.Processing.Filters.Experimental(
                contrast_distortion=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(
                    correction=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
                        enabled=False
                    )
                )
            )
        ],
    )


def test_settings_processing_filters_experimental(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="experimental",
        value=zivid.Settings.Processing.Filters.Experimental(),
        expected_data_type=zivid.Settings.Processing.Filters.Experimental,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Experimental,
        [
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(
                removal=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
                    enabled=False
                )
            )
        ],
        [
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(
                removal=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
                    enabled=True
                )
            )
        ],
    )


def test_settings_processing_filters_experimental_contrast_distortion(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental(),
        member="contrast_distortion",
        value=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(),
        expected_data_type=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion,
        [
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
                enabled=True
            ),
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
                enabled=False
            ),
        ],
        [
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(
                enabled=True
            ),
            zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(
                enabled=True
            ),
        ],
    )


def test_settings_processing_filters_experimental_contrast_distortion_removal(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal,
        [True],
        [False],
    )


def test_settings_processing_filters_experimental_contrast_distortion_removal_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_processing_filters_experimental_contrast_distortion_removal_threshold(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Removal(),
        member="threshold",
        value=0.4,
        expected_data_type=float,
    )


def test_settings_processing_filters_experimental_contrast_distortion_correction(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion(),
        member="correction",
        value=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(),
        expected_data_type=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction,
        [True],
        [False],
    )


def test_settings_processing_filters_experimental_contrast_distortion_correction_strength(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(),
        member="strength",
        value=0.59,
        expected_data_type=float,
    )


def test_settings_processing_filters_experimental_contrast_distortion_correction_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Experimental.ContrastDistortion.Correction(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_noise(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="noise",
        value=zivid.Settings.Processing.Filters.Noise(),
        expected_data_type=zivid.Settings.Processing.Filters.Noise,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Noise,
        [zivid.Settings.Processing.Filters.Noise.Removal(enabled=True)],
        [zivid.Settings.Processing.Filters.Noise.Removal(enabled=False)],
    )


def test_settings_processing_filters_noise_removal(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Noise.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Noise.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Noise.Removal, [True], [False],
    )


def test_settings_processing_filters_noise_removal_enabled(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise.Removal(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_noise_removal_threshold(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise.Removal(),
        member="threshold",
        value=50.0,
        expected_data_type=float,
    )


def test_settings_processing_filters_outlier(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="outlier",
        value=zivid.Settings.Processing.Filters.Outlier(),
        expected_data_type=zivid.Settings.Processing.Filters.Outlier,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Outlier,
        [zivid.Settings.Processing.Filters.Outlier.Removal(enabled=True)],
        [zivid.Settings.Processing.Filters.Outlier.Removal(enabled=False)],
    )


def test_settings_processing_filters_outlier_removal(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Outlier.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Outlier.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Outlier.Removal, [True], [False],
    )


def test_settings_processing_filters_outlier_removal_enabled(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier.Removal(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_outlier_removal_threshold(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier.Removal(),
        member="threshold",
        value=89,
        expected_data_type=float,
    )


def test_settings_processing_filters_reflection(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="reflection",
        value=zivid.Settings.Processing.Filters.Reflection(),
        expected_data_type=zivid.Settings.Processing.Filters.Reflection,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Reflection,
        [zivid.Settings.Processing.Filters.Reflection.Removal(enabled=True)],
        [zivid.Settings.Processing.Filters.Reflection.Removal(enabled=False)],
    )


def test_settings_processing_filters_reflection_removal(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Reflection(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Reflection.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Reflection.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Reflection.Removal, [True], [False],
    )


def test_settings_processing_filters_reflection_removal_enabled(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Reflection.Removal(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_processing_filters_smoothing(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="smoothing",
        value=zivid.Settings.Processing.Filters.Smoothing(),
        expected_data_type=zivid.Settings.Processing.Filters.Smoothing,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Smoothing,
        [zivid.Settings.Processing.Filters.Smoothing.Gaussian(enabled=True)],
        [zivid.Settings.Processing.Filters.Smoothing.Gaussian(enabled=False)],
    )


def test_settings_processing_filters_smoothing_gaussian(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing(),
        member="gaussian",
        value=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        expected_data_type=zivid.Settings.Processing.Filters.Smoothing.Gaussian,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Smoothing.Gaussian, [True], [False],
    )


def test_settings_processing_filters_smoothing_gaussian_sigma(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        member="sigma",
        value=1.798888,
        expected_data_type=float,
    )


def test_settings_processing_filters_smoothing_gaussian_enabled(application,):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_print_acquisition(application):
    import zivid

    print(zivid.Settings.Acquisition())


def test_print_processing(application):
    import zivid

    print(zivid.Settings.Processing())
