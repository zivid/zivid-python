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
    common_to_internal_generation,
)
import tempfile
from pathlib import Path
import inflection
from _zivid import CameraState


def start_traverse():
    common_to_internal_generation(
        internal_class=CameraState, settings_type="CameraState"
    )
