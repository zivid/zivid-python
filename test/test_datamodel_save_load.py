import pytest


@pytest.mark.parametrize(
    "datamodel",
    [
        "Settings",
        "Settings2D",
        "CameraState",
        "CameraInfo",
        "FrameInfo",
        "capture_assistant.SuggestSettingsParameters",
    ],
)
def test_basic_save_load(application, datamodel_yml_dir, datamodel):
    from operator import attrgetter
    from pathlib import Path
    from tempfile import TemporaryDirectory
    import zivid

    filename = datamodel.split(".")[-1] + ".yml"
    load_path = datamodel_yml_dir / filename

    # Load original file
    original = attrgetter(datamodel)(zivid).load(load_path)
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / filename
        # Save to new file
        original.save(save_path)
        # Load the new file back and check
        loaded = attrgetter(datamodel)(zivid).load(save_path)
        assert original == loaded
        assert str(original) == str(loaded)
        # These should be different from default-constructed
        attrgetter(datamodel)(zivid)().save(save_path)
        default = attrgetter(datamodel)(zivid).load(save_path)
        assert default != loaded
        assert str(default) != str(loaded)
