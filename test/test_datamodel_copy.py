import copy
from operator import attrgetter

import pytest
import zivid


def _is_deep_copy(a, b):
    if isinstance(a, (list, tuple, set)):
        return all(_is_deep_copy(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return all(k in b and _is_deep_copy(v, b[k]) for k, v in a.items())
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        return all(k in b.__dict__ and _is_deep_copy(v, b.__dict__[k]) for k, v in a.__dict__.items())
    if a is None:
        return b is None
    return a is not b


@pytest.mark.parametrize(
    "copy_func",
    [
        copy.copy,
        copy.deepcopy,
    ],
)
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
def test_datamodel_copy(
    application,  # pylint: disable=unused-argument
    datamodel_yml_dir,
    datamodel,
    copy_func,
):

    filename = datamodel.split(".")[-1] + ".yml"
    load_path = datamodel_yml_dir / filename

    cls = attrgetter(datamodel)(zivid)

    default = cls()
    original = cls.load(load_path)
    assert original != default

    default_copy = copy_func(default)
    original_copy = copy_func(original)

    assert original_copy is not original
    assert original_copy == original
    assert default_copy is not default
    assert default_copy == default
    assert original_copy is not default_copy
    assert original_copy != default_copy

    if copy_func is copy.copy:
        assert not _is_deep_copy(default, default_copy)
        assert not _is_deep_copy(original, original_copy)
    elif copy_func is copy.deepcopy:
        assert _is_deep_copy(default, default_copy)
        assert _is_deep_copy(original, original_copy)
