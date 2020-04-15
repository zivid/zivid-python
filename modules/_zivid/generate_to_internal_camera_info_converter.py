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
from _zivid import CameraInfo


def start_traverse():
    common_to_internal_generation(internal_class=CameraInfo, settings_type="CameraInfo")


# def _create_to_internal_converter(node_data, settings_type: str):
#     temp_internal_name = "internal_{name}".format(
#         name=inflection.underscore(node_data.name)
#     )
#     nested_converters = [
#         _create_to_internal_converter(element, settings_type=settings_type)
#         for element in node_data.children
#     ]
#     nested_converters_string = "\n".join(nested_converters)
#     convert_member_logic = ""
#     if node_data.member_variables:
#         for member in node_data.member_variables:
#             convert_member_logic += "\n    if {name}.{member} is not None:\n".format(
#                 name=inflection.underscore(node_data.name),
#                 member=inflection.underscore(member),
#             )
#
#             convert_member_logic += "\n        {temp_internal_name}.{member} = _zivid.Settings2D{path}".format(
#                 temp_internal_name=temp_internal_name,
#                 member=inflection.underscore(member),
#                 path=".{path}.{member_as_class}({name}.{member})".format(
#                     path=node_data.path,
#                     member_as_class=member,
#                     name=inflection.underscore(node_data.name),
#                     member=inflection.underscore(member),
#                 )
#                 if node_data.path
#                 else "()",
#             )
#             convert_member_logic += "\n    else:"
#             convert_member_logic += "\n        {temp_internal_name}.{member} = _zivid.Settings2D{path}".format(
#                 temp_internal_name=temp_internal_name,
#                 member=inflection.underscore(member),
#                 path=".{path}.{member_as_class}()".format(
#                     path=node_data.path, member_as_class=member,
#                 )
#                 if node_data.path
#                 else "()",
#             )
#
#     convert_children_logic = ""
#     if node_data.children:
#         for child in node_data.children:
#             convert_children_logic += "\n    {temp_internal_name}.{child} = _to_internal_{child}({name}.{child})".format(
#                 temp_internal_name=temp_internal_name,
#                 child=inflection.underscore(child.name),
#                 name=inflection.underscore(node_data.name),
#             )
#     else:
#         convert_children_logic = "pass # no children"
#
#     base_class = """
# def _to_internal_{target_name}({target_name}):
#     {temp_internal_name} = _zivid.Settings2D{path}
#     {nested_converters}
#     {convert_member_logic}
#     {convert_children_logic}
#     return {temp_internal_name}
# """.format(
#         target_name=inflection.underscore(node_data.name),
#         nested_converters=nested_converters_string,
#         convert_member_logic=convert_member_logic,
#         convert_children_logic=convert_children_logic,
#         path="." + node_data.path + "()" if node_data.path else "()",
#         temp_internal_name=temp_internal_name,
#     )
#     indented_lines = list()
#     for line in base_class.splitlines():
#         indented_lines.append("    " + line)
#     return "\n".join(indented_lines)
#
