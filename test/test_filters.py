def test_default_init_filters(application):  # pylint: disable=unused-argument
    import zivid

    filters = zivid.Settings.Filters()
    assert filters.contrast
    assert filters.outlier
    assert filters.saturated
    assert filters.reflection
    assert filters.gaussian
    assert isinstance(filters.contrast, zivid.Settings.Filters.Contrast)
    assert isinstance(filters.outlier, zivid.Settings.Filters.Outlier)
    assert isinstance(filters.saturated, zivid.Settings.Filters.Saturated)
    assert isinstance(filters.reflection, zivid.Settings.Filters.Reflection)
    assert isinstance(filters.gaussian, zivid.Settings.Filters.Gaussian)


def test_default_init_contrast(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    contrast = zivid.Settings.Filters.Contrast()
    assert contrast.enabled is not None
    assert contrast.threshold is not None
    assert isinstance(contrast.enabled, bool)
    assert isinstance(contrast.threshold, numbers.Real)


def test_default_init_outlier(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    outlier = zivid.Settings.Filters.Outlier()
    assert outlier.enabled is not None
    assert outlier.threshold is not None
    assert isinstance(outlier.enabled, bool)
    assert isinstance(outlier.threshold, numbers.Real)


def test_default_init_saturated(application):  # pylint: disable=unused-argument
    import zivid

    saturated = zivid.Settings.Filters.Saturated()
    assert saturated.enabled is not None
    assert isinstance(saturated.enabled, bool)


def test_default_init_reflection(application):  # pylint: disable=unused-argument
    import zivid

    reflection = zivid.Settings.Filters.Reflection()
    assert reflection.enabled is not None
    assert isinstance(reflection.enabled, bool)


def test_default_init_gaussian(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    gaussian = zivid.Settings.Filters.Gaussian()
    assert gaussian.enabled is not None
    assert gaussian.sigma is not None
    assert isinstance(gaussian.enabled, bool)
    assert isinstance(gaussian.sigma, numbers.Real)


def test_init_contrast(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    enabled = True
    threshold = 123

    contrast = zivid.Settings.Filters.Contrast(enabled=enabled, threshold=threshold)
    assert contrast.enabled == enabled
    assert isinstance(contrast.enabled, bool)
    assert contrast.threshold == threshold
    assert isinstance(contrast.threshold, numbers.Real)


def test_init_outlier(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    enabled = True
    threshold = 12345

    outlier = zivid.Settings.Filters.Outlier(enabled=enabled, threshold=threshold)
    assert outlier.enabled == enabled
    assert isinstance(outlier.enabled, bool)
    assert outlier.threshold == threshold
    assert isinstance(outlier.threshold, numbers.Real)


def test_init_saturated(application):  # pylint: disable=unused-argument
    import zivid

    enabled = True

    saturated = zivid.Settings.Filters.Saturated(enabled=enabled)
    assert saturated.enabled == enabled
    assert isinstance(saturated.enabled, bool)


def test_init_reflection(application):  # pylint: disable=unused-argument
    import zivid

    enabled = True

    reflection = zivid.Settings.Filters.Reflection(enabled=enabled)
    assert reflection.enabled == enabled
    assert isinstance(reflection.enabled, bool)


def test_init_gaussian(application):  # pylint: disable=unused-argument
    import numbers
    import zivid

    enabled = True
    sigma = 1234

    gaussian = zivid.Settings.Filters.Gaussian(enabled=enabled, sigma=sigma)
    assert gaussian.enabled == enabled
    assert isinstance(gaussian.enabled, bool)
    assert gaussian.sigma == sigma
    assert isinstance(gaussian.sigma, numbers.Real)


def test_equal_filters(application):  # pylint: disable=unused-argument
    import zivid

    filters1 = zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=True))
    filters2 = zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=True))

    assert filters1 == filters2


def test_not_equal_filters(application):  # pylint: disable=unused-argument
    import zivid

    filters1 = zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=True))
    filters2 = zivid.Settings.Filters(zivid.Settings.Filters.Contrast(enabled=False))

    assert filters1 != filters2


def test_equal_contrast(application):  # pylint: disable=unused-argument
    import zivid

    contrast1 = zivid.Settings.Filters.Contrast(enabled=True)
    contrast2 = zivid.Settings.Filters.Contrast(enabled=True)

    assert contrast1 == contrast2


def test_not_equal_contrast(application):  # pylint: disable=unused-argument
    import zivid

    contrast1 = zivid.Settings.Filters.Contrast(enabled=True)
    contrast2 = zivid.Settings.Filters.Contrast(enabled=False)

    assert contrast1 != contrast2


def test_equal_outlier(application):  # pylint: disable=unused-argument
    import zivid

    outlier1 = zivid.Settings.Filters.Outlier(enabled=True)
    outlier2 = zivid.Settings.Filters.Outlier(enabled=True)

    assert outlier1 == outlier2


def test_not_equal_outlier(application):  # pylint: disable=unused-argument
    import zivid

    outlier1 = zivid.Settings.Filters.Outlier(enabled=True)
    outlier2 = zivid.Settings.Filters.Outlier(enabled=False)

    assert outlier1 != outlier2


def test_equal_saturated(application):  # pylint: disable=unused-argument
    import zivid

    saturated1 = zivid.Settings.Filters.Saturated(enabled=True)
    saturated2 = zivid.Settings.Filters.Saturated(enabled=True)

    assert saturated1 == saturated2


def test_not_equal_saturated(application):  # pylint: disable=unused-argument
    import zivid

    saturated1 = zivid.Settings.Filters.Saturated(enabled=True)
    saturated2 = zivid.Settings.Filters.Saturated(enabled=False)

    assert saturated1 != saturated2


def test_equal_reflection(application):  # pylint: disable=unused-argument
    import zivid

    reflection1 = zivid.Settings.Filters.Reflection(enabled=True)
    reflection2 = zivid.Settings.Filters.Reflection(enabled=True)

    assert reflection1 == reflection2


def test_not_equal_reflection(application):  # pylint: disable=unused-argument
    import zivid

    reflection1 = zivid.Settings.Filters.Reflection(enabled=True)
    reflection2 = zivid.Settings.Filters.Reflection(enabled=False)

    assert reflection1 != reflection2


def test_equal_gaussian(application):  # pylint: disable=unused-argument
    import zivid

    gaussian1 = zivid.Settings.Filters.Gaussian(enabled=True)
    gaussian2 = zivid.Settings.Filters.Gaussian(enabled=True)

    assert gaussian1 == gaussian2


def test_not_equal_gaussian(application):  # pylint: disable=unused-argument
    import zivid

    gaussian1 = zivid.Settings.Filters.Gaussian(enabled=True)
    gaussian2 = zivid.Settings.Filters.Gaussian(enabled=False)

    assert gaussian1 != gaussian2
