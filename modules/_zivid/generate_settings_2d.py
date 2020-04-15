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
    from _zivid._zivid import Settings2D

    common_class_generation(
        internal_class=Settings2D,
        settings_type="Settings2D",
        converter_import="_settings_2d_converter",
    )
