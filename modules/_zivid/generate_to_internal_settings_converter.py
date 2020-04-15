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
from _zivid._zivid import Settings


def start_traverse():
    common_to_internal_generation(internal_class=Settings, settings_type="Settings")


#
# def start_traverse():
#
#
#     # from zivid import Settings
#
#     data_model = _recursion(InternalSettings, indentation_level=0)
#     with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
#         temp_file = Path(temp_file.name)
#         raw_text = _imports(internal=True, settings=False)
#         raw_text += create_to_internal_converter(data_model, settings_type="Settings")
#
#         new_lines = []
#         for line in raw_text.splitlines():
#             new_lines.append(line[4:])
#
#         temp_file.write_text("\n".join(new_lines))
#         print(temp_file.read_text())
#         subprocess.check_output((f"black {temp_file}"), shell=True)
#         print(temp_file.read_text())
#         path_to_settings = (
#             Path(__file__).resolve() / ".." / ".." / "zivid" / "settings__3d.py"
#         ).resolve()
#         path_to_settings.write_text(temp_file.read_text())
#
