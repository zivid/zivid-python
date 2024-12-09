import pytest


def test_default_settings(application):
    import zivid

    settings = zivid.Settings()

    assert settings.color is None

    assert isinstance(settings.acquisitions, list)
    assert len(settings.acquisitions) == 0
    assert settings.engine is None
    assert isinstance(settings.diagnostics, zivid.Settings.Diagnostics)
    assert settings.diagnostics.enabled is None
    assert isinstance(settings.processing, zivid.Settings.Processing)
    assert isinstance(settings.processing.color, zivid.Settings.Processing.Color)
    assert isinstance(
        settings.processing.color.balance, zivid.Settings.Processing.Color.Balance
    )
    assert settings.processing.color.gamma is None
    assert settings.processing.color.balance.red is None
    assert settings.processing.color.balance.green is None
    assert settings.processing.color.balance.blue is None
    assert settings.processing.color.experimental.mode is None

    assert isinstance(settings.processing.filters, zivid.Settings.Processing.Filters)
    assert isinstance(
        settings.processing.filters.experimental,
        zivid.Settings.Processing.Filters.Experimental,
    )

    assert isinstance(
        settings.processing.filters.cluster, zivid.Settings.Processing.Filters.Cluster
    )
    assert isinstance(
        settings.processing.filters.cluster.removal,
        zivid.Settings.Processing.Filters.Cluster.Removal,
    )

    assert settings.processing.filters.cluster.removal.enabled is None
    assert settings.processing.filters.cluster.removal.min_area is None
    assert settings.processing.filters.cluster.removal.max_neighbor_distance is None

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
        settings.processing.filters.hole.repair,
        zivid.Settings.Processing.Filters.Hole.Repair,
    )

    assert settings.processing.filters.hole.repair.enabled is None
    assert settings.processing.filters.hole.repair.hole_size is None
    assert settings.processing.filters.hole.repair.strictness is None

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
    assert settings.processing.filters.reflection.removal.mode is None

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

    assert isinstance(settings.region_of_interest, zivid.Settings.RegionOfInterest)
    assert isinstance(
        settings.region_of_interest.box, zivid.Settings.RegionOfInterest.Box
    )
    assert isinstance(
        settings.region_of_interest.depth, zivid.Settings.RegionOfInterest.Depth
    )

    assert settings.region_of_interest.box.enabled is None
    assert settings.region_of_interest.box.extents is None
    assert settings.region_of_interest.box.point_o is None
    assert settings.region_of_interest.box.point_a is None
    assert settings.region_of_interest.box.point_b is None
    assert settings.region_of_interest.depth.enabled is None
    assert settings.region_of_interest.depth.range is None


def test_set_color_settings():
    # pylint: disable=protected-access

    from zivid import Settings, Settings2D
    from zivid.settings import _to_settings, _to_internal_settings

    settings = Settings()
    assert settings.color is None

    to_cpp_and_back = _to_settings(_to_internal_settings(settings))
    assert to_cpp_and_back.color is None
    assert to_cpp_and_back == settings

    settings.color = Settings2D()
    assert settings.color is not None
    assert isinstance(settings.color, Settings2D)
    assert settings.color == Settings2D()
    to_cpp_and_back = _to_settings(_to_internal_settings(settings))
    assert to_cpp_and_back.color is not None
    assert to_cpp_and_back == settings

    settings = Settings(color=Settings2D())
    assert settings.color is not None
    assert isinstance(settings.color, Settings2D)
    assert settings.color == Settings2D()
    to_cpp_and_back = _to_settings(_to_internal_settings(settings))
    assert to_cpp_and_back.color is not None
    assert to_cpp_and_back == settings

    settings = Settings(color=Settings2D(acquisitions=(Settings2D.Acquisition(),)))
    assert settings.color is not None
    assert isinstance(settings.color, Settings2D)
    assert len(settings.color.acquisitions) == 1
    to_cpp_and_back = _to_settings(_to_internal_settings(settings))
    assert to_cpp_and_back.color is not None
    assert to_cpp_and_back == settings

    settings = Settings()
    settings.color = Settings2D()
    settings.color.acquisitions.append(Settings2D.Acquisition())
    assert len(settings.color.acquisitions) == 1
    to_cpp_and_back = _to_settings(_to_internal_settings(settings))
    assert len(to_cpp_and_back.color.acquisitions) == 1
    assert to_cpp_and_back == settings


def test_set_acquisition_list():
    from zivid import Settings

    settings = Settings()

    settings.acquisitions = [
        Settings.Acquisition(gain=1.0),
        Settings.Acquisition(gain=2.0),
        Settings.Acquisition(gain=3.0),
    ]
    assert len(settings.acquisitions) == 3
    assert settings.acquisitions is not None
    assert isinstance(settings.acquisitions, list)
    for element in settings.acquisitions:
        assert isinstance(element, Settings.Acquisition)

    assert settings.acquisitions[0].gain == 1.0
    assert settings.acquisitions[1].gain == 2.0
    assert settings.acquisitions[2].gain == 3.0

    settings.acquisitions[0].gain = 4.0
    assert settings.acquisitions[0].gain == 4.0


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


def test_settings_diagnostics(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings(),
        member="diagnostics",
        value=zivid.Settings.Diagnostics(),
        expected_data_type=zivid.Settings.Diagnostics,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Diagnostics,
        [True],
        [False],
    )


def test_settings_diagnostics_enabled(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Diagnostics(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_engine(application):
    import zivid

    for value in zivid.Settings.Engine.valid_values():
        pytest.helpers.set_attribute_tester(
            settings_instance=zivid.Settings(),
            member="engine",
            value=value,
            expected_data_type=str,
        )
    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings(),
        member="engine",
        value=None,
        expected_data_type=type(None),
    )
    # Is optional enum
    zivid.Settings(engine="stripe")
    zivid.Settings(engine="phase")
    zivid.Settings(engine=None)
    with pytest.raises(KeyError):
        zivid.Settings(engine="_dummy_")


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


def test_settings_processing_color_gamma(
    application,
):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color(),
        member="gamma",
        value=0.85,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color(),
        member="balance",
        value=zivid.Settings.Processing.Color.Balance(),
        expected_data_type=zivid.Settings.Processing.Color.Balance,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color.Balance,
        [1.1, 1.1, 1.1],
        [1.2, 1.1, 1.1],
    )


def test_settings_processing_color_experimental(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color(),
        member="experimental",
        value=zivid.Settings.Processing.Color.Experimental(),
        expected_data_type=zivid.Settings.Processing.Color.Experimental,
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color.Experimental,
        ["automatic"],
        [None],
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color.Experimental,
        ["automatic"],
        ["useFirstAcquisition"],
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Color.Experimental,
        ["automatic"],
        ["toneMapping"],
    )


def test_settings_processing_color_experimental_mode(application):
    import zivid

    for value in zivid.Settings.Processing.Color.Experimental.Mode.valid_values():
        pytest.helpers.set_attribute_tester(
            settings_instance=zivid.Settings.Processing.Color.Experimental(),
            member="mode",
            value=value,
            expected_data_type=str,
        )
    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Experimental(),
        member="mode",
        value=None,
        expected_data_type=type(None),
    )
    # Is optional enum
    zivid.Settings.Processing.Color.Experimental(mode="automatic")
    zivid.Settings.Processing.Color.Experimental(mode="useFirstAcquisition")
    zivid.Settings.Processing.Color.Experimental(mode="toneMapping")
    zivid.Settings.Processing.Color.Experimental(mode=None)
    with pytest.raises(KeyError):
        zivid.Settings.Processing.Color.Experimental(mode="_dummy_")


def test_settings_processing_color_balance_red(
    application,
):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="red",
        value=2,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_green(
    application,
):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="green",
        value=1.5,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_color_balance_blue(
    application,
):
    import zivid
    import numbers

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Color.Balance(),
        member="blue",
        value=1,
        expected_data_type=numbers.Real,
    )


def test_settings_processing_filters(
    application,
):
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
            zivid.Settings.Processing.Filters.Cluster(
                zivid.Settings.Processing.Filters.Cluster.Removal(enabled=True)
            )
        ],
        [
            zivid.Settings.Processing.Filters.Cluster(
                zivid.Settings.Processing.Filters.Cluster.Removal(enabled=False)
            )
        ],
    )


def test_settings_processing_filters_experimental(
    application,
):
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


def test_settings_processing_filters_experimental_contrast_distortion(
    application,
):
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


def test_settings_processing_filters_hole_repair(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Hole(),
        member="repair",
        value=zivid.Settings.Processing.Filters.Hole.Repair(),
        expected_data_type=zivid.Settings.Processing.Filters.Hole.Repair,
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Hole.Repair,
        [True],
        [False],
    )


def test_settings_processing_filters_hole_repair_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Hole.Repair(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_processing_filters_hole_repair_holesize(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Hole.Repair(),
        member="hole_size",
        value=0.3,
        expected_data_type=float,
    )


def test_settings_processing_filters_hole_repair_strictness(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Hole.Repair(),
        member="strictness",
        value=3,
        expected_data_type=int,
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


def test_settings_processing_filters_cluster(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters(),
        member="cluster",
        value=zivid.Settings.Processing.Filters.Cluster(),
        expected_data_type=zivid.Settings.Processing.Filters.Cluster,
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Cluster,
        [zivid.Settings.Processing.Filters.Cluster.Removal(enabled=True)],
        [zivid.Settings.Processing.Filters.Cluster.Removal(enabled=False)],
    )


def test_settings_processing_filters_cluster_removal(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Cluster(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Cluster.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Cluster.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Cluster.Removal,
        [True],
        [False],
    )


def test_settings_processing_filters_cluster_removal_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Cluster.Removal(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_cluster_removal_minarea(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Cluster.Removal(),
        member="min_area",
        value=150.0,
        expected_data_type=float,
    )


def test_settings_processing_filters_cluster_removal_maxneighbordistance(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Cluster.Removal(),
        member="max_neighbor_distance",
        value=5.0,
        expected_data_type=float,
    )


def test_settings_processing_filters_noise(
    application,
):
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


def test_settings_processing_filters_noise_removal(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Noise.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Noise.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Noise.Removal,
        [True],
        [False],
    )


def test_settings_processing_filters_noise_removal_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise.Removal(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_noise_removal_threshold(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Noise.Removal(),
        member="threshold",
        value=50.0,
        expected_data_type=float,
    )


def test_settings_processing_filters_outlier(
    application,
):
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


def test_settings_processing_filters_outlier_removal(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Outlier.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Outlier.Removal,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Outlier.Removal,
        [True],
        [False],
    )


def test_settings_processing_filters_outlier_removal_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier.Removal(),
        member="enabled",
        value=False,
        expected_data_type=bool,
    )


def test_settings_processing_filters_outlier_removal_threshold(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Outlier.Removal(),
        member="threshold",
        value=89,
        expected_data_type=float,
    )


def test_settings_processing_filters_reflection(
    application,
):
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


def test_settings_processing_filters_reflection_removal(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Reflection(),
        member="removal",
        value=zivid.Settings.Processing.Filters.Reflection.Removal(),
        expected_data_type=zivid.Settings.Processing.Filters.Reflection.Removal,
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Reflection.Removal,
        [True],
        [False],
    )

    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Reflection.Removal,
        [
            True,
            zivid.Settings.Processing.Filters.Reflection.Removal.Mode.global_,
        ],
        [
            True,
            zivid.Settings.Processing.Filters.Reflection.Removal.Mode.local,
        ],
    )


def test_settings_processing_filters_reflection_removal_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Reflection.Removal(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_processing_filters_reflection_removal_mode(
    application,
):
    import zivid

    for (
        value
    ) in zivid.Settings.Processing.Filters.Reflection.Removal.Mode.valid_values():
        pytest.helpers.set_attribute_tester(
            settings_instance=zivid.Settings.Processing.Filters.Reflection.Removal(),
            member="mode",
            value=value,
            expected_data_type=str,
        )
    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Reflection.Removal(),
        member="mode",
        value=None,
        expected_data_type=type(None),
    )

    # Is optional enum
    zivid.Settings.Processing.Filters.Reflection.Removal(mode="global")
    zivid.Settings.Processing.Filters.Reflection.Removal(mode="local")
    with pytest.raises(KeyError):
        zivid.Settings.Processing.Filters.Reflection.Removal(mode="_dummy_")


def test_settings_processing_filters_smoothing(
    application,
):
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


def test_settings_processing_filters_smoothing_gaussian(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing(),
        member="gaussian",
        value=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        expected_data_type=zivid.Settings.Processing.Filters.Smoothing.Gaussian,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.Processing.Filters.Smoothing.Gaussian,
        [True],
        [False],
    )


def test_settings_processing_filters_smoothing_gaussian_sigma(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        member="sigma",
        value=1.798888,
        expected_data_type=float,
    )


def test_settings_processing_filters_smoothing_gaussian_enabled(
    application,
):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.Processing.Filters.Smoothing.Gaussian(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_regionofinterest(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings(),
        member="region_of_interest",
        value=zivid.Settings.RegionOfInterest(),
        expected_data_type=zivid.Settings.RegionOfInterest,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.RegionOfInterest,
        [zivid.Settings.RegionOfInterest.Box(enabled=True)],
        [zivid.Settings.RegionOfInterest.Box(enabled=False)],
    )


def test_settings_regionofinterest_box(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest(),
        member="box",
        value=zivid.Settings.RegionOfInterest.Box(),
        expected_data_type=zivid.Settings.RegionOfInterest.Box,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.RegionOfInterest.Box,
        [True],
        [False],
    )


def test_settings_regionofinterest_box_enabled(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Box(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_regionofinterest_box_pointo(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Box(),
        member="point_o",
        value=[100.0, 200.0, 300.0],
        expected_data_type=list,
    )


def test_settings_regionofinterest_box_pointa(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Box(),
        member="point_a",
        value=[100.0, 200.0, 300.0],
        expected_data_type=list,
    )


def test_settings_regionofinterest_box_pointb(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Box(),
        member="point_b",
        value=[100.0, 200.0, 300.0],
        expected_data_type=list,
    )


def test_settings_regionofinterest_box_extents(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Box(),
        member="extents",
        value=[-100.0, 500.0],
        expected_data_type=list,
    )

    # A range must have two elements
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Box(extents=[])
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Box(extents=[100.0])
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Box(extents=[100.0, 200.0, 300.0])

    # A range must have min <= max
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Box(extents=[100.0, 50.0])


def test_settings_regionofinterest_depth(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest(),
        member="depth",
        value=zivid.Settings.RegionOfInterest.Depth(),
        expected_data_type=zivid.Settings.RegionOfInterest.Depth,
    )
    pytest.helpers.equality_tester(
        zivid.Settings.RegionOfInterest.Depth,
        [True],
        [False],
    )


def test_settings_regionofinterest_depth_enabled(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Depth(),
        member="enabled",
        value=True,
        expected_data_type=bool,
    )


def test_settings_regionofinterest_depth_range(application):
    import zivid

    pytest.helpers.set_attribute_tester(
        settings_instance=zivid.Settings.RegionOfInterest.Depth(),
        member="range",
        value=[100.0, 1500.0],
        expected_data_type=list,
    )

    # A range must have two elements
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Depth(range=[])
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Depth(range=[100.0])
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Depth(range=[100.0, 200.0, 300.0])

    # A range must have min <= max
    with pytest.raises(TypeError):
        zivid.Settings.RegionOfInterest.Depth(range=[100.0, -200.0])


def test_print_settings(application):
    import zivid

    print(zivid.Settings())


def test_print_acquisition(application):
    import zivid

    print(zivid.Settings.Acquisition())


def test_print_processing(application):
    import zivid

    print(zivid.Settings.Processing())
