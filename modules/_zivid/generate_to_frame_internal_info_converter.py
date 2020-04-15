from _zivid.common import (
    _create_class,
    _imports,
    _recursion,
    common_to_internal_generation,
)
import tempfile
from pathlib import Path
import inflection
from _zivid import FrameInfo


def start_traverse():
    common_to_internal_generation(internal_class=FrameInfo, settings_type="FrameInfo")
