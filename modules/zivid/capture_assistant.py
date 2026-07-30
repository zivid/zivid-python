"""Contains the Capture Assistant functionality."""

from __future__ import annotations

import _zivid
from zivid._suggest_settings_parameters import (
    SuggestSettingsParameters,
    _to_internal_capture_assistant_suggest_settings_parameters,
)
from zivid.camera import Camera
from zivid.settings import Settings, _to_settings


def suggest_settings(camera: Camera, suggest_settings_parameters: SuggestSettingsParameters) -> Settings:
    """Find settings for the current scene based on given parameters.

    The suggested settings returned from this function should be passed into
    camera.capture() to capture and retrieve the Frame containing a point cloud.

    Args:
        camera: A Camera instance
        suggest_settings_parameters: A SuggestSettingsParameters instance

    Returns:
        A Settings instance optimized for the current scene
    """
    internal_settings = _zivid.capture_assistant.suggest_settings(
        camera._Camera__impl,  # pylint: disable=protected-access
        _to_internal_capture_assistant_suggest_settings_parameters(suggest_settings_parameters),
    )
    return _to_settings(internal_settings)
