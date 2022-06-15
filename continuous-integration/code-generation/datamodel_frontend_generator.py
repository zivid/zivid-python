import argparse
import inspect
import subprocess
import re
import tempfile
from typing import Any, List, Optional, Sequence, Tuple
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent, indent

import inflection
import _zivid


def _to_snake_case(name: str) -> str:
    # Workaround due to flaw in inflection.underscore(): Substitute any "X_d" with "Xd"
    return re.sub(r"(\d)_d", r"\g<1>d", inflection.underscore(name)).lower()


def _indent(content: str) -> str:
    return indent(content, " " * 4)


def _get_dot_path(base_type: str, node_path: Path) -> str:
    if node_path == Path("."):
        return f"_zivid.{base_type}"
    return f"_zivid.{base_type}.{'.'.join(node_path.parts)}"


def _get_underscore_name(base_type: str, node_path: Path) -> str:
    underscored_base_type = _to_snake_case(base_type).replace(".", "_")
    if node_path == Path("."):
        return underscored_base_type
    return f"{underscored_base_type}_{'_'.join([_to_snake_case(part) for part in node_path.parts])}"


@dataclass
class MemberInfo:
    is_optional: bool
    underlying_type: str


@dataclass
class ContainerInfo:
    contained_type: str
    underlying_type: str


@dataclass
class NodeData:
    # pylint: disable=too-many-instance-attributes
    name: str
    snake_case: str
    is_leaf: bool
    is_enum: bool
    is_uninstantiated_node: bool
    enum_vars: tuple
    path: Path
    children: Tuple
    member_variables: Tuple["NodeData", ...]
    member_containers: Tuple["NodeData", ...]
    underlying_zivid_class: Any
    member_info: Optional[MemberInfo] = None
    container_info: Optional[ContainerInfo] = None


def _inner_classes_list(cls: Any) -> List:
    return [
        cls_attribute
        for cls_attribute in cls.__dict__.values()
        if inspect.isclass(cls_attribute)
    ]


def _imports(extra_imports: Sequence[str]) -> str:

    linter_exceptions = [
        "too-many-lines",
        "protected-access",
        "too-few-public-methods",
        "too-many-arguments",
        "line-too-long",
        "missing-function-docstring",
        "missing-class-docstring",
        "too-many-branches",
        "too-many-boolean-expressions",
    ]

    header = ""
    header += "'''Auto generated, do not edit.'''\n"
    header += f"# pylint: disable={','.join(linter_exceptions)}\n"
    if extra_imports:
        header += "\n".join([f"import {module}" for module in extra_imports]) + "\n"
    header += "import _zivid"
    return header


def _create_init_special_member_function(node_data: NodeData, base_type: str) -> str:

    full_dot_path = _get_dot_path(base_type, node_data.path)
    signature_vars = ""
    member_variable_set = ""

    for container in node_data.member_containers:

        if container.container_info is None:
            raise RuntimeError(f"Unexpected lack of container info: {container}")

        signature_vars += f"{container.snake_case}=None,"
        underlying_type = container.container_info.underlying_type
        member_variable_set += dedent(
            f"""
            if {container.snake_case} is None:
                self._{container.snake_case} = []
            elif isinstance({container.snake_case}, {container.container_info.underlying_type} ):
                self._{container.snake_case} = []
                for item in {container.snake_case}:
                    if isinstance(item, self.{container.container_info.contained_type}):
                        self._{container.snake_case}.append(item)
                    else:
                        raise TypeError('Unsupported type {{item_type}}'.format(item_type=type(item)))
            else:
                raise TypeError(
                    'Unsupported type, expected: {underlying_type} or None, got {{value_type}}'.format(value_type=type({container.snake_case}))
                )
            """
        )

    for member in node_data.member_variables:

        if member.member_info is None:
            raise RuntimeError(f"Unexpected lack of member info: {member}")

        if member.member_info.is_optional:
            is_none_check = f"or {member.snake_case} is None"
            none_message = " or None"
        else:
            is_none_check = ""
            none_message = ""
        signature_vars += f"{member.snake_case}={full_dot_path}.{member.name}().value,"

        if member.is_enum:

            member_variable_set += dedent(
                f"""
                if isinstance({member.snake_case},{full_dot_path}.{member.name}.enum) {is_none_check}:
                    self._{member.snake_case} = {full_dot_path}.{member.name}({member.snake_case})
                elif isinstance({member.snake_case}, str):
                    self._{member.snake_case} = {full_dot_path}.{member.name}(self.{member.name}._valid_values[{member.snake_case}])
                else:
                    raise TypeError('Unsupported type, expected: str{none_message}, got {{value_type}}'.format(value_type=type({member.snake_case})))
                """
            )

        else:
            underlying_type = member.member_info.underlying_type
            member_variable_set += dedent(
                f"""
                if isinstance({member.snake_case}, {member.member_info.underlying_type} ) {is_none_check}:
                    self._{member.snake_case} = {full_dot_path}.{member.name}({member.snake_case})
                else:
                    raise TypeError(
                        'Unsupported type, expected: {underlying_type}{none_message}, got {{value_type}}'.format(value_type=type({member.snake_case}))
                    )
                """
            )

    for child_class in node_data.children:

        if child_class.is_uninstantiated_node:
            continue
        signature_vars += f"{child_class.snake_case}=None,"

        member_variable_set += dedent(
            f"""
            if {child_class.snake_case} is None:
                {child_class.snake_case} = self.{child_class.name}()
            if not isinstance({child_class.snake_case}, self.{child_class.name}):
                raise TypeError('Unsupported type: {{value}}'.format(value=type({child_class.snake_case})))
            self._{child_class.snake_case} = {child_class.snake_case}
            """
        )

    return dedent(
        """
        def __init__(self, {signature_vars}):
            {member_variable_set}
        """
    ).format(
        signature_vars=signature_vars, member_variable_set=_indent(member_variable_set)
    )


def _create_eq_special_member_function(node_data: NodeData) -> str:
    member_variables_equality = []
    for container in node_data.member_containers:
        member_variables_equality.append(
            f"self._{container.snake_case} == other._{container.snake_case}"
        )
    for member in node_data.member_variables:
        member_variables_equality.append(
            f"self._{member.snake_case} == other._{member.snake_case}"
        )

    for child in node_data.children:
        if child.is_uninstantiated_node:
            continue
        member_variables_equality.append(
            f"self._{child.snake_case} == other._{child.snake_case}"
        )
    equality_logic = " and ".join(member_variables_equality)
    return dedent(
        f"""
        def __eq__(self, other):
            if (
                {equality_logic}
            ):
                return True
            return False
        """
    )


def _create_str_special_member_function(node_data: NodeData, base_type: str) -> str:

    full_underscore_path = _get_underscore_name(base_type, node_data.path)
    str_content = f"str(_to_internal_{full_underscore_path}(self))"

    return dedent(
        f"""
        def __str__(self):
            return {str_content}
        """
    )


def _create_properties(node_data: NodeData, base_type: str) -> str:

    get_properties = "\n"
    set_properties = "\n"

    for node in node_data.member_containers:

        if node.container_info is None:
            raise RuntimeError(f"Unexpected lack of container info: {node}")

        get_properties += dedent(
            f"""
            @property
            def {node.snake_case}(self):
                return self._{node.snake_case}
            """
        )

        set_properties += dedent(
            f"""
            @{node.snake_case}.setter
            def {node.snake_case}(self,value):
                if not isinstance(value, {node.container_info.underlying_type}):
                    raise TypeError('Unsupported type {{value}}'.format(value=type(value)))
                self._{node.snake_case} = []
                for item in value:
                    if isinstance(item, self.{node.container_info.contained_type}):
                        self._{node.snake_case}.append(item)
                    else:
                        raise TypeError('Unsupported type {{item_type}}'.format(item_type=type(item)))
            """
        )

    for member in node_data.member_variables:

        if member.member_info is None:
            raise RuntimeError(f"Unexpected lack of member info: {member}")

        full_dot_path = _get_dot_path(base_type, node_data.path)

        if member.member_info.is_optional:
            is_none_check = "or value is None"
            can_be_none_error_message_part = " or None"
        else:
            is_none_check = ""
            can_be_none_error_message_part = ""

        if member.is_enum:
            expected_types_str = "str" + can_be_none_error_message_part

            get_properties += dedent(
                """
                @property
                def {enum_member}(self):
                    if self._{enum_member}.value is None:
                        return None
                    for key, internal_value in self.{enum_class}._valid_values.items():
                        if internal_value == self._{enum_member}.value:
                           return key
                    raise ValueError("Unsupported value {{value}}".format(value=self._{enum_member}))
                """
            ).format(enum_member=member.snake_case, enum_class=member.name)

            set_properties += dedent(
                f"""
                @{member.snake_case}.setter
                def {member.snake_case}(self, value):
                    if isinstance(value, str):
                        self._{member.snake_case} = {full_dot_path}.{member.name}(self.{member.name}._valid_values[value])
                    elif isinstance(value, {full_dot_path}.{member.name}.enum) {is_none_check}:
                        self._{member.snake_case} = {full_dot_path}.{member.name}(value)
                    else:
                        raise TypeError('Unsupported type, expected: {expected_types_str}, got {{value_type}}'.format(value_type=type(value)))
                """
            )

        else:
            underlying_type_str = member.member_info.underlying_type[1:-2].split(",")
            expected_types_str = (
                " or ".join(underlying_type_str) + can_be_none_error_message_part
            )

            get_properties += dedent(
                """
                @property
                def {member}(self):
                    return self._{member}.value
                """
            ).format(member=member.snake_case)

            set_properties += dedent(
                f"""
                @{member.snake_case}.setter
                def {member.snake_case}(self,value):
                    if isinstance(value,{member.member_info.underlying_type}) {is_none_check}:
                        self._{member.snake_case} = {full_dot_path}.{member.name}(value)
                    else:
                        raise TypeError('Unsupported type, expected: {expected_types_str}, got {{value_type}}'.format(value_type=type(value)))
                """
            )

    for child in node_data.children:
        if child.is_uninstantiated_node:
            continue

        get_properties += dedent(
            """
            @property
            def {member}(self):
                return self._{member}
            """
        ).format(member=child.snake_case)

        set_properties += dedent(
            f"""
            @{child.snake_case}.setter
            def {child.snake_case}(self,value):
                if not isinstance(value, self.{child.name}):
                    raise TypeError('Unsupported type {{value}}'.format(value=type(value)))
                self._{child.snake_case} = value
            """
        )

    return dedent(
        """
        {get_properties}
        {set_properties}
        """
    ).format(
        get_properties=get_properties,
        set_properties=set_properties,
    )


def _create_save_load_functions(node_data: NodeData, base_type: str):
    full_dot_path = _get_dot_path(base_type=base_type, node_path=node_data.path)
    underscore_name = _get_underscore_name(
        base_type=base_type, node_path=node_data.path
    )
    return dedent(
        f"""
        @classmethod
        def load(cls, file_name):
            return _to_{underscore_name}({full_dot_path}(str(file_name)))

        def save(self, file_name):
            _to_internal_{underscore_name}(self).save(str(file_name))
        """
    )


def _create_enum_class(member: NodeData, base_type: str) -> str:

    full_dot_path = _get_dot_path(base_type, member.path)

    static_members = ["\n"]
    valid_values = ["\n"]
    for enum_var in member.enum_vars:
        # Some members have a trailing underscore added to avoid collision with reserved keywords.
        # For the string-representation we strip that trailing underscore out.
        static_members.append(f'{enum_var} = "{enum_var.rstrip("_")}"')
        valid_values.append(f'"{enum_var.rstrip("_")}": {full_dot_path}.{enum_var},')

    return dedent(
        """
        class {name}:

            {static_members}

            _valid_values = {{{valid_values}}}

            @classmethod
            def valid_values(cls):
                return list(cls._valid_values.keys())
        """
    ).format(
        name=member.name,
        static_members=_indent("\n".join(static_members)),
        valid_values=_indent("\n".join(valid_values)),
    )


def _create_enum_classes(node_data: NodeData, base_type: str) -> str:
    return "\n".join(
        [
            _create_enum_class(member, base_type)
            for member in node_data.member_variables
            if member.is_enum
        ]
    )


def _create_class(node_data: NodeData, base_type: str, is_root: bool) -> str:
    nested_classes = [
        _create_class(element, base_type=base_type, is_root=False)
        for element in node_data.children
    ]
    nested_classes_string = "\n".join(nested_classes)

    return dedent(
        """
        class {class_name}:
            {nested_classes}
            {enum_classes}
            {init_function}
            {get_set_properties}
            {save_load_functions}
            {eq_function}
            {str_function}
        """
    ).format(
        class_name=node_data.name,
        nested_classes=_indent(nested_classes_string),
        enum_classes=_indent(_create_enum_classes(node_data, base_type=base_type)),
        init_function=_indent(
            _create_init_special_member_function(node_data, base_type=base_type)
        ),
        eq_function=_indent(_create_eq_special_member_function(node_data)),
        str_function=_indent(
            _create_str_special_member_function(node_data, base_type=base_type)
        ),
        get_set_properties=_indent(_create_properties(node_data, base_type=base_type)),
        save_load_functions=_indent(
            _create_save_load_functions(node_data, base_type=base_type)
        )
        if is_root
        else "",
    )


def _parse_internal_datamodel(current_class: Any) -> NodeData:
    child_classes = []
    if hasattr(current_class, "valid_values") and hasattr(current_class, "enum"):
        is_leaf = True
    elif current_class.node_type.name == "leaf_data_model_list":
        is_leaf = False
    else:
        for my_cls in _inner_classes_list(current_class):
            child_classes.append(_parse_internal_datamodel(my_cls))
        is_leaf = not bool(_inner_classes_list(current_class))

    member_variables = []
    member_containers = []
    to_be_removed = []
    for child in child_classes:

        if child.underlying_zivid_class.node_type.name == "leaf_data_model_list":
            child.container_info = ContainerInfo(
                contained_type=child.underlying_zivid_class.contained_type,
                underlying_type=child.underlying_zivid_class.value_type,
            )
            member_containers.append(child)
            to_be_removed.append(child)

        elif child.is_leaf:
            child.member_info = MemberInfo(
                is_optional=child.underlying_zivid_class.is_optional,
                underlying_type=child.underlying_zivid_class.value_type,
            )
            member_variables.append(child)
            to_be_removed.append(child)
    child_classes = [
        element for element in child_classes if element not in to_be_removed
    ]

    if hasattr(current_class, "valid_values") and hasattr(current_class, "enum"):
        is_enum_class = True
        enum_vars = []
        for member in dir(current_class.enum):
            if str(member).startswith("__"):
                continue
            if str(member) == "name" or str(member) == "value":
                continue
            enum_vars.append(member)

    else:
        is_enum_class = False
        enum_vars = []

    return NodeData(
        name=current_class.name,
        snake_case=_to_snake_case(current_class.name),
        is_leaf=is_leaf,
        is_enum=is_enum_class,
        is_uninstantiated_node=current_class.uninstantiated_node,
        enum_vars=tuple(enum_vars),
        path=Path(current_class.path),
        children=tuple(child_classes),
        member_variables=tuple(member_variables),
        member_containers=tuple(member_containers),
        underlying_zivid_class=current_class,
    )


def _create_to_frontend_converter(node_data: NodeData, base_type: str) -> str:

    base_typename = base_type.split(".")[-1]
    temp_internal_name = f"internal_{node_data.snake_case}"
    nested_converters = [
        _create_to_frontend_converter(element, base_type=base_type)
        for element in node_data.children
    ]
    underscored_path = _get_underscore_name(base_type, node_data.path)

    container_convert_logic = ""
    for container in node_data.member_containers:
        if container.container_info is None:
            raise RuntimeError(f"Unexpected lack of container info: {container}")

        contained_converter = f"_to_{underscored_path}_{_to_snake_case(container.container_info.contained_type)}"
        container_convert_logic += f"{container.snake_case} = [{contained_converter}(value) for value in {temp_internal_name}.{container.snake_case}.value],"

    member_convert_logic = ""
    for member in node_data.member_variables:
        member_convert_logic += (
            "{member} = {temp_internal_name}.{member}.value,".format(
                member=member.snake_case,
                temp_internal_name=temp_internal_name,
            )
        )

    child_convert_logic = ""
    for child in node_data.children:
        if not child.is_uninstantiated_node:
            child_convert_logic += (
                "{child_name}=_to_{child}({temp_internal_name}.{child_name}),".format(
                    child_name=child.snake_case,
                    child=f"{underscored_path}_{child.snake_case}",
                    temp_internal_name=temp_internal_name,
                )
            )

    base_function = dedent(
        f"""
        def _to_{underscored_path}(internal_{node_data.snake_case}):
            return {'.'.join((base_typename,) + node_data.path.parts)}({container_convert_logic} {child_convert_logic} {member_convert_logic})
        """
    )
    nested_converters_string = "\n".join(nested_converters)
    return dedent(
        f"""
        {nested_converters_string}
        {base_function}
        """
    )


def _create_to_internal_converter(node_data: NodeData, base_type: str) -> str:

    temp_internal_name = f"internal_{node_data.snake_case}"
    nested_converters = [
        _create_to_internal_converter(element, base_type=base_type)
        for element in node_data.children
    ]
    underscored_path = _get_underscore_name(base_type, node_data.path)
    full_dot_path = _get_dot_path(base_type, node_data.path)

    convert_member_logic = ""

    for container in node_data.member_containers:
        if container.container_info is None:
            raise RuntimeError(f"Unexpected lack of container info: {container}")

        converter_for_contained = f"_to_internal_{underscored_path}_{_to_snake_case(container.container_info.contained_type)}"
        convert_member_logic += dedent(
            f"""
            temp_{container.snake_case} = {full_dot_path}.{container.name}()
            for value in {node_data.snake_case}.{container.snake_case}:
                temp_{container.snake_case}.append({converter_for_contained}(value))
            {temp_internal_name}.{container.snake_case} = temp_{container.snake_case}
            """
        )

    if node_data.member_variables:
        for member in node_data.member_variables:
            constructor_arg = (
                f"{node_data.snake_case}._{member.snake_case}.value"
                if member.is_enum
                else f"{node_data.snake_case}.{member.snake_case}"
            )
            convert_member_logic += f"\n{temp_internal_name}.{member.snake_case} = {full_dot_path}.{member.name}({constructor_arg})"

    convert_children_logic = ""
    if node_data.children:
        for child in node_data.children:
            if not child.is_uninstantiated_node:
                convert_children_logic += f"\n{temp_internal_name}.{child.snake_case} = _to_internal_{underscored_path}_{child.snake_case}({node_data.snake_case}.{child.snake_case})"

    base_function = dedent(
        """
        def {function_name}({function_arg}):
            {temp_internal_name} = {path}()
            {convert_member_logic}
            {convert_children_logic}
            return {temp_internal_name}
        """
    ).format(
        function_name=f"_to_internal_{underscored_path}",
        function_arg=node_data.snake_case,
        temp_internal_name=temp_internal_name,
        path=full_dot_path,
        convert_member_logic=_indent(convert_member_logic),
        convert_children_logic=_indent(convert_children_logic),
    )

    nested_converters_string = "\n".join(nested_converters)
    return dedent(
        f"""
        {nested_converters_string}
        {base_function}
        """
    )


def _print_data_model(data_model: NodeData, indentation_level: int) -> None:
    if indentation_level == 0:
        print("*" * 70)
    indentation = " " * 4 * indentation_level
    print(f"{indentation}{data_model.name}")
    for member in data_model.member_variables:
        print(f"{indentation}    - {member.snake_case}")
    for child in data_model.children:
        _print_data_model(child, indentation_level + 1)


def _get_submodule(internal_class: Any) -> str:
    base_module = inspect.getmodule(internal_class)
    if base_module is None:
        raise RuntimeError(f"Failed to detect module of: {internal_class}")
    return base_module.__name__.replace("_zivid._zivid", "")


def _generate_datamodel_frontend(
    internal_class: Any,
    destination: Path,
    extra_imports: List[str],
    verbose: bool = False,
):
    # Parse the _zivid datamodel into a NodeData tree
    data_model = _parse_internal_datamodel(internal_class)
    _print_data_model(data_model, 0)

    # Convert NodeData tree to source code
    base_type = f"{_get_submodule(internal_class)}.{internal_class.name}".lstrip(".")
    raw_text = _imports(extra_imports=extra_imports)
    raw_text += _create_class(data_model, base_type=base_type, is_root=True)
    raw_text += _create_to_frontend_converter(data_model, base_type=base_type)
    raw_text += _create_to_internal_converter(data_model, base_type=base_type)

    # Format source code and save to destination path
    with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:

        temp_file_path = Path(temp_file.name)
        temp_file_path.write_text(raw_text, encoding="utf-8")
        if verbose:
            print(temp_file_path.read_text(encoding="utf-8"))
        subprocess.check_output((f"black {temp_file_path}"), shell=True)
        if verbose:
            print(temp_file_path.read_text(encoding="utf-8"))
        destination.write_text(
            temp_file_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"Saved {internal_class.name} to {destination}.")


def generate_all_datamodels(dest_dir: Path) -> None:

    for internal_class, filename, extra_imports in [
        (_zivid.Settings, "settings.py", ["datetime", "collections.abc"]),
        (_zivid.Settings2D, "settings_2d.py", ["datetime", "collections.abc"]),
        (_zivid.CameraInfo, "camera_info.py", []),
        (_zivid.CameraState, "camera_state.py", []),
        (_zivid.FrameInfo, "frame_info.py", ["datetime"]),
        (
            _zivid.capture_assistant.SuggestSettingsParameters,
            "_suggest_settings_parameters.py",
            ["datetime"],
        ),
        (_zivid.CameraIntrinsics, "camera_intrinsics.py", []),
    ]:
        _generate_datamodel_frontend(
            internal_class=internal_class,
            destination=dest_dir / filename,
            extra_imports=extra_imports,
        )


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest-dir", required=True, type=Path, help="Destination directory"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _args()
    generate_all_datamodels(dest_dir=args.dest_dir)
