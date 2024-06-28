"""Contains the Capture Assistant functionality."""

import _zivid

from zivid._suggest_settings_parameters import (  # pylint: disable=unused-import
    SuggestSettingsParameters,
)
from zivid._suggest_settings_parameters import (
    _to_internal_capture_assistant_suggest_settings_parameters,
)
from zivid.settings import _to_settings


def suggest_settings(camera, suggest_settings_parameters):
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
        _to_internal_capture_assistant_suggest_settings_parameters(
            suggest_settings_parameters
        ),
    )
    return _to_settings(internal_settings)
