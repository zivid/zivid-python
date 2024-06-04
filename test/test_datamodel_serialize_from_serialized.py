import pytest


@pytest.mark.parametrize(
    "datamodel",
    [
        "Settings",
        "Settings2D",
        "CameraState",
        "CameraInfo",
        "FrameInfo",
        "NetworkConfiguration",
        "capture_assistant.SuggestSettingsParameters",
    ],
)
def test_serialize_from_serialized(application, datamodel_yml_dir, datamodel):
    from operator import attrgetter

    import zivid

    filename = datamodel.split(".")[-1] + ".yml"
    load_path = datamodel_yml_dir / filename

    cls = attrgetter(datamodel)(zivid)

    default = cls()
    original = cls.load(load_path)
    assert original != default

    original_yaml = original.serialize()
    default_yaml = default.serialize()
    assert isinstance(original_yaml, str)
    assert isinstance(default_yaml, str)
    assert original_yaml != default_yaml

    original_deserialized = cls.from_serialized(original_yaml)
    default_deserialized = cls.from_serialized(default_yaml)
    assert original_deserialized == original
    assert default_deserialized == default
    assert original_deserialized != default_deserialized
