import pytest


def test_3d_presets(application):  # pylint: disable=unused-argument
    import zivid

    for model in zivid.CameraInfo.Model.valid_values():
        if model in [
            zivid.CameraInfo.Model.zividOnePlusLarge,
            zivid.CameraInfo.Model.zividOnePlusMedium,
            zivid.CameraInfo.Model.zividOnePlusSmall,
        ]:
            with pytest.raises(
                RuntimeError,
                match=f"Internal error: The camera model '{model}' is not supported by this version of Zivid SDK.",
            ):
                zivid.presets.categories(model)
        else:
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


def test_2d_presets(application):  # pylint: disable=unused-argument
    import zivid

    for model in zivid.CameraInfo.Model.valid_values():
        if model in [
            zivid.CameraInfo.Model.zividOnePlusLarge,
            zivid.CameraInfo.Model.zividOnePlusMedium,
            zivid.CameraInfo.Model.zividOnePlusSmall,
        ]:
            with pytest.raises(
                RuntimeError,
                match=f"Internal error: The camera model '{model}' is not supported by this version of Zivid SDK.",
            ):
                zivid.presets.categories2d(model)
        else:
            categories = zivid.presets.categories2d(model)
            assert categories
            for category in categories:
                assert isinstance(category, zivid.presets.Category)
                assert category.name != ""
                assert category.presets
                for preset in category.presets:
                    assert isinstance(preset, zivid.presets.Preset)
                    assert preset.name != ""
                    assert isinstance(preset.settings, zivid.Settings2D)
