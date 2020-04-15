"""This file imports used classes, modules and packages."""
"""This file imports used classes, modules and packages."""
import inspect
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple
import subprocess
from _zivid.common import (
    _create_class,
    _imports,
    _recursion,
    common_class_generation,
)


def start_traverse():
    from _zivid._zivid import CameraState

    common_class_generation(
        internal_class=CameraState,
        settings_type="CameraState",
        converter_import="_camera_state_converter",
    )
