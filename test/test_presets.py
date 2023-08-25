def test_presets(application):  # pylint: disable=unused-argument
    import zivid

    for model in zivid.CameraInfo.Model.valid_values():
        categories = zivid.presets.categories(model)
        assert categories
        for category in categories:
            assert isinstance(category, zivid.presets.Category)
            assert category.name != ""
            assert category.presets
            for preset in category.presets:
                assert isinstance(preset, zivid.presets.Preset)
                assert preset.name != ""
                assert isinstance(preset.settings, zivid.Settings)
