import inspect
import tempfile
from typing import List, Tuple, Optional
from dataclasses import dataclass
import inspect
from collections import namedtuple
from pathlib import Path
import subprocess
import inflection


@dataclass
class MemberVariable:
    camel_case: str
    snake_case: str
    # variable_name: str
    default_value: str
    underlying_type: str
    is_optional: bool


@dataclass
class UnderlyingLeafValue:
    is_optional: bool
    value: str


@dataclass
class NodeData:
    name: str
    is_leaf: bool
    is_enum: bool
    enum_vars: tuple
    enum_default_value: str
    path: str
    children: tuple
    member_variables: Tuple[MemberVariable]
    indentation_level: int
    _zivid_class: str  # the current class
    # leaf_underlying_type: Optional[UnderlyingLeafValue]


def _inner_classes_list(cls) -> List:
    return [
        cls_attribute
        for cls_attribute in cls.__dict__.values()
        if inspect.isclass(cls_attribute)
    ]


def _imports(
    internal: bool, settings: bool, additional_imports: tuple = tuple()
) -> str:
    imports = "    '''Auto generated, do not edit'''\n"
    imports += "    import datetime\n"
    imports += "    import types\n"
    imports += "    import collections.abc\n"
    if internal:
        imports += "    import _zivid\n"
    if settings:
        imports += "    import zivid\n"
    for additional_import in additional_imports:
        imports += f"    import {additional_import}"
    return imports


def _get_member_variables(node_data, settings_type: str):
    member_variables = []
    if node_data.member_variables:
        for member_var in node_data.member_variables:
            print(member_var)
            default_value = f"_zivid.{settings_type}().{f'{node_data.path}().' if node_data.path else ''}{member_var.camel_case}().value"
            # camel_case = member_var
            # snake_case
            # variable_name = member_var.snake_case#inflection.underscore(member_var)
            print("this is current class when getting member variables:")
            print()
            member_variables.append(
                MemberVariable(
                    camel_case=member_var.camel_case,
                    default_value=default_value,
                    snake_case=member_var.snake_case,
                    underlying_type=member_var.underlying_type,
                    is_optional=member_var.is_optional,
                )
            )
    print(member_variables)
    return member_variables


def _get_child_class_member_variables(node_data):
    child_class_members_variables = []
    if node_data.children:
        for child in node_data.children:
            child_class_members_variables.append(
                MemberVariable(
                    camel_case=child.name,
                    snake_case=inflection.underscore(child.name),
                    # default_value=f"{child.name}()",
                    default_value="None",
                    underlying_type=None,
                    is_optional=True,
                )
            )

    return child_class_members_variables


def _variable_names(node_data, settings_type: str):
    member_variables = _get_member_variables(node_data, settings_type)
    child_class_member_variables = _get_child_class_member_variables(node_data)
    variable_names = list()
    for member in member_variables:
        variable_names.append(member.snake_case)
    for child_class in child_class_member_variables:
        variable_names.append(child_class.snake_case)
    return variable_names


def _create_init_special_member_function(node_data, settings_type: str):
    if node_data.is_enum:
        return "\n    def __init__(self,value=none):\n        self._value = value\n"
    member_variables = _get_member_variables(node_data, settings_type)
    child_class_member_variables = _get_child_class_member_variables(node_data)
    signature_vars = ""
    member_variable_set = ""
    path = ".{path}".format(path=node_data.path,) if node_data.path else ""
    for member in member_variables:
        # is_optional_str_start =
        if member.is_optional:
            # expected_types = member.underlying_type
            is_none_check = f"or {member.snake_case} is None"
            can_be_none_error_message_part = " or None"
        else:
            is_none_check = ""
            can_be_none_error_message_part = ""
            # expected_types = member.underlying_type
        signature_vars += f"{member.snake_case}={member.default_value},"
        member_variable_set += f"\n        if isinstance({member.snake_case},{member.underlying_type}) {is_none_check}:"
        # member_variable_set += f"\n        if {member.snake_case} is not None:"
        member_variable_set += f"\n            self._{member.snake_case} = _zivid.{settings_type}{path}.{member.camel_case}({member.snake_case})"
        member_variable_set += f"\n        else:"
        member_variable_set += f"\n            raise TypeError('Unsupported type, expected: {member.underlying_type}{can_be_none_error_message_part}, got {{value_type}}'.format(value_type=type({member.snake_case})))"
        # member_variable_set += f"\n        else:"
        # member_variable_set += f"\n            self._{member.snake_case} = _zivid.{settings_type}{path}.{member.camel_case}()"

    for child_class in child_class_member_variables:
        signature_vars += f"{child_class.snake_case}={child_class.default_value},"
        member_variable_set += f"\n        if {child_class.snake_case} is None:"
        member_variable_set += f"\n            {child_class.snake_case} = zivid.{settings_type}{path}.{child_class.camel_case}()"
        member_variable_set += f"\n        if not isinstance({child_class.snake_case}, zivid.{settings_type}{path}.{child_class.camel_case}):"
        member_variable_set += f"\n            raise TypeError('Unsupported type: {{value}}'.format(value=type({child_class.snake_case})))"
        member_variable_set += (
            f"\n        self._{child_class.snake_case} = {child_class.snake_case}"
        )

    # for variable_name in _variable_names(node_data, settings_type):
    #    path = ".{path}".format(path=node_data.path,) if node_data.path else ""
    #    member_variable_set += f"\n        if {variable_name} is not None:"
    #    member_variable_set += f"\n            self._{variable_name} = _zivid.{settings_type}{path}({variable_name})"

    return """
    def __init__(
        self,
        {signature_vars}
    ):
        {member_variable_set}""".format(
        signature_vars=signature_vars, member_variable_set=member_variable_set
    )


def _create_eq_special_member_function(node_data, settings_type: str):
    if node_data.is_enum:
        return "\n    def __eq__(self,other):\n        if self.value == other.value:\n            return True\n        return False"
    member_variables_equality = list()
    member_variables = _get_member_variables(node_data, settings_type)
    child_class_member_variables = _get_child_class_member_variables(node_data)
    for member in member_variables:
        member_variables_equality.append(
            f"self._{member.snake_case} == other._{member.snake_case}"
        )
    for child in child_class_member_variables:
        member_variables_equality.append(
            f"self._{child.snake_case} == other._{child.snake_case}"
        )
    equality_logic = " and ".join(member_variables_equality)
    return """def __eq__(self, other):
        if (
            {equality_logic}
        ):
            return True
        return False""".format(
        equality_logic=equality_logic
    )


def _create_str_special_member_function(node_data, settings_type: str):
    str_content = ""
    member_variables = _get_member_variables(node_data, settings_type)
    # for member in member_variables:
    #     pass
    standard_path = inflection.underscore(node_data.path.replace(".", "_"))
    if not standard_path:
        output = inflection.underscore(node_data.name)
    else:
        output = standard_path
    to_internal_function = f"to_internal_"  # _{node_data.name}"
    to_internal = (
        f"{inflection.underscore(settings_type)}_{inflection.underscore(node_data.path).replace('.', '_')}"
        if node_data.path
        else f"{inflection.underscore(settings_type)}"  # {inflection.underscore(node_data.name)}
    )
    str_content = f"str(zivid._{inflection.underscore(settings_type).replace('.', '_')}_converter.{to_internal_function}{to_internal}(self))"
    # str_content

    return """def __str__(self):
            return {str_content}""".format(
        str_content=str_content
    )


##
## def _create_str_special_member_function(node_data, settings_type: str):
##     member_variables_str = "    "
##     formatting_string = ""
##
##     member_variables = _get_member_variables(node_data, settings_type)
##     child_class_member_variables = _get_child_class_member_variables(node_data)
##     for member in member_variables:
##         element = member.snake_case
##         member_variables_str += f"{element}: {{{element}}}\n    "
##         formatting_string += "{variable_name}=self.{variable_name},".format(
##             variable_name=element
##         )
##
##     for child in child_class_member_variables:
##         element = child.snake_case
##         member_variables_str += f"{element}: {{{element}}}\n    "
##         formatting_string += "{variable_name}=self.{variable_name},".format(
##             variable_name=element
##         )
##
##     member_variables_str.strip()
##     str_content = """'''{name}:
## {member_variables_str}'''.format({formatting_string})""".format(
##         name=node_data.name,
##         member_variables_str=member_variables_str,
##         formatting_string=formatting_string,
##     )
##     return """def __str__(self):
##             return {str_content}""".format(
##         str_content=str_content
##     )
##
##         if member.is_optional:
##             #expected_types = member.underlying_type
##             is_none_check = f"or {member.snake_case} is None"
##             can_be_none_error_message_part = " or None"
##         else:
##             is_none_check = ""
##             can_be_none_error_message_part=""
##             #expected_types = member.underlying_type
##         signature_vars += f"{member.snake_case}={member.default_value},"
##         member_variable_set += f"\n        if isinstance({member.snake_case},{member.underlying_type}) {is_none_check}:"
##         # member_variable_set += f"\n        if {member.snake_case} is not None:"
##         member_variable_set += f"\n            self._{member.snake_case} = _zivid.{settings_type}{path}.{member.camel_case}({member.snake_case})"
##         member_variable_set += f"\n        else:"
##         member_variable_set += f"\n            raise TypeError('Unsupported type, expected: {member.underlying_type}{can_be_none_error_message_part}, got {{value_type}}'.format(value_type=type({member.snake_case})))"


def _leaf_underlying_type():
    return "some_type"


def _create_properties(node_data, settings_type: str):
    get_properties = "\n"
    set_properties = "\n"

    get_member_property_template = (
        "    @property\n    def {member}(self):\n        return self._{member}.value\n"
    )
    set_member_property_template = "    @{member}.setter\n    def {member}(self,value):\n        if isinstance(value,{expected_types}) {none_check}:\n            self._{member} = _zivid.{settings_type}{path}.{non_snake_member}(value)\n        else:\n            raise TypeError('Unsupported type, expected: {expected_types_str}, got {{value_type}}'.format(value_type=type(value)))\n"
    for member in node_data.member_variables:
        path = ".{path}".format(path=node_data.path,) if node_data.path else ""
        get_properties += get_member_property_template.format(member=member.snake_case)
        if member.is_optional:
            # expected_types = member.underlying_type
            is_none_check = "or value is None"
            can_be_none_error_message_part = " or None"
        else:
            is_none_check = ""
            can_be_none_error_message_part = ""
        expected_types_str = " or ".join(member.underlying_type[1:-2].split(","))
        expected_types_str += can_be_none_error_message_part
        set_properties += set_member_property_template.format(
            member=member.snake_case,
            path=path,
            non_snake_member=member.camel_case,
            settings_type=settings_type,
            expected_types=member.underlying_type,
            none_check=is_none_check,
            can_be_none_error_message_part=can_be_none_error_message_part,
            expected_types_str=expected_types_str,
        )
    set_child_property_template = "    @{member}.setter\n    def {member}(self,value):\n        if not isinstance(value, zivid.{settings_type}{path}.{non_snake_member}):\n            raise TypeError('Unsupported type {{value}}'.format(value=type(value)))\n        self._{member} = value\n"
    get_child_property_template = (
        "    @property\n    def {member}(self):\n        return self._{member}\n"
    )
    for child in node_data.children:
        path = ".{path}".format(path=node_data.path,) if node_data.path else ""
        get_properties += get_child_property_template.format(
            member=inflection.underscore(child.name)
        )
        set_properties += set_child_property_template.format(
            member=inflection.underscore(child.name),
            path=path,
            non_snake_member=child.name,
            settings_type=settings_type,
        )
    return f"{get_properties}\n{set_properties}"


def _create_class_variables(node_data):
    sub_strings = []
    for enum_key, enum_value in node_data.enum_vars:
        sub_strings.append(
            f"{enum_key} = _zivid.capture_assistant.SuggestSettingsParameters.{node_data.path}.enum.{enum_value}"
        )  # {enum_value}")
    return "\n    ".join(
        sub_strings
    )  # + f"# _zivid.capture_assistant.{node_data.path}.enum"


def _create_class(node_data, settings_type: str):
    nested_classes = [
        _create_class(element, settings_type=settings_type)
        for element in node_data.children
    ]
    nested_classes_string = "\n".join(nested_classes)
    base_class = """
class {class_name}:
    {nested_classes}
    {class_variables}
    {init_special_member_function}
    {get_set_properties}
    {eq_special_member_function}
    {str_special_member_function}
""".format(
        class_name=node_data.name,
        nested_classes=nested_classes_string,
        class_variables=_create_class_variables(node_data),
        init_special_member_function=_create_init_special_member_function(
            node_data, settings_type=settings_type
        ),
        eq_special_member_function=_create_eq_special_member_function(
            node_data, settings_type=settings_type
        ),
        str_special_member_function=_create_str_special_member_function(
            node_data, settings_type=settings_type
        ),
        get_set_properties=_create_properties(node_data, settings_type=settings_type),
    )
    indented_lines = list()
    for line in base_class.splitlines():
        indented_lines.append("    " + line)
    return "\n".join(indented_lines)


def _recursion(current_class, indentation_level, parent_class=None):
    child_classes = list()
    print(f"this is class: {current_class.name}")
    print(dir(current_class))
    if not (hasattr(current_class, "valid_values") and hasattr(current_class, "enum")):
        for my_cls in _inner_classes_list(current_class):
            child_classes.append(
                _recursion(
                    my_cls,
                    indentation_level=indentation_level + 1,
                    parent_class=current_class,
                )
            )
        is_leaf = not bool(_inner_classes_list(current_class))
    elif not hasattr(current_class, "_zivid_class"):
        is_leaf = True
    elif current_class._zivid_class.node_type in (
        "NodeType.leaf_value",
        "NodeType.leaf_data_model_list",
    ):
        is_leaf = True
    else:
        is_leaf = False

    member_variables = list()
    to_be_removed = list()
    for child in child_classes:
        if child.is_leaf:
            # print(current_class.name)
            # print(type(child.name))
            # if is_leaf:
            # print("This is leaf")
            # try:
            # print(dir(current_class))
            path = current_class.path.replace("/", ".")
            is_enum = True
            # if current_class.is_optional():
            #     is_optional = True
            # else:
            #     is_optional = False
            # # except RuntimeError as ex:
            # #     print(ex)
            # #     is_optional = "Failed"
            # #try:
            print(child.name)
            print(child._zivid_class.node_type)
            print(dir(child))
            print(dir(child._zivid_class))
            # if child._zivid_class.node_type != "NodeType.leaf_data_model_list":
            leaf_underlying_type = child._zivid_class.value_type()
            print("found type!!!")
            print(leaf_underlying_type)
            # # except RuntimeError as ex:
            # #     print(ex)
            # #     leaf_underlying_type = "Failed to obtain type"
            # leaf_underlying_value = UnderlyingLeafValue(is_optional=is_optional, value = leaf_underlying_type)
            # # else:
            # #     leaf_underlying_value = None
            # print(f"this is leaf_underlying_type: {leaf_underlying_value}")

            member_variables.append(
                MemberVariable(
                    camel_case=child.name,
                    snake_case=inflection.underscore(child.name),
                    default_value="None",
                    # underlying_type=leaf_underlying_value
                    underlying_type=leaf_underlying_type,
                    is_optional=child._zivid_class.is_optional(),
                )
            )
            to_be_removed.append(child)
    child_classes = [
        element for element in child_classes if element not in to_be_removed
    ]
    # print(current_class.name)
    # print(dir(current_class))

    if hasattr(current_class, "valid_values") and hasattr(current_class, "enum"):

        # print("this is a enum thingy")

        # print("defaultvalue:")
        # print(current_class().value)
        # print(dir(current_class().value))
        # print(current_class().value.name)
        is_enum_class = True
        path = current_class.path.replace("/", ".")
        enum_default_value = current_class().value.name
        enum_vars = []
        members = [a for a in dir(current_class.enum)]
        for member in members:
            if str(member).startswith("__"):
                continue
            if str(member) == "name":
                continue
            # print("this is member: " + member)
            # print(getattr(current_class.enum, member).name)
            enum_vars.append((member, getattr(current_class.enum, member).name))

    else:
        path = current_class.path.replace("/", ".")
        is_enum_class = False
        # print("this is not a enum thingy")
        enum_vars = []
        enum_default_value = None

    # print(enum_vars)
    # print(is_enum)

    print("\n------------------------------------------------")

    my_class = NodeData(
        name=current_class.name,
        is_leaf=is_leaf,
        is_enum=is_enum_class,
        enum_vars=enum_vars,
        enum_default_value=enum_default_value,
        path=path,
        children=child_classes,
        member_variables=member_variables,
        indentation_level=indentation_level,
        _zivid_class=current_class,
        # leaf_underlying_type=leaf_underlying_value,
    )
    return my_class
    # temp_internal_name = "internal_{path}{name}".format(
    #     name=inflection.underscore(node_data.name),
    #     path=f"{node_data.path.replace('.', '_')}_" if node_data.path else ""
    # )


def create_to_not_internal_converter(node_data, settings_type: str):
    temp_internal_name = "internal_{name}".format(
        name=inflection.underscore(node_data.name)
    )
    nested_converters = [
        create_to_not_internal_converter(element, settings_type=settings_type)
        for element in node_data.children
    ]
    nested_converters_string = "\n".join(nested_converters)
    return_class = "zivid.{settings_type}{path}".format(
        temp_internal_name=temp_internal_name,
        settings_type=settings_type,
        path=".{path}".format(path=node_data.path,) if node_data.path else "",
    )
    member_convert_logic = ""
    for member in node_data.member_variables:
        member_convert_logic += "{member} = {temp_internal_name}.{member}.value,".format(
            member=member.snake_case,
            # member_not_snake_case=member.lower(),
            temp_internal_name=temp_internal_name,
        )

    # global_functions = ""
    child_convert_logic = ""
    for child in node_data.children:
        child_convert_logic += "{child_name}=to_{child}({temp_internal_name}.{child_name}),".format(
            child_name=inflection.underscore(child.name),
            child=f"{inflection.underscore(settings_type)}_{inflection.underscore(node_data.path).replace('.', '_')}_{inflection.underscore(child.name)}"
            if node_data.path
            else f"{inflection.underscore(settings_type)}_{inflection.underscore(child.name)}",
            # child_not_snake_case=child.name.lower(),
            temp_internal_name=temp_internal_name,
        )
        # global_functions += "\n    global to{path}{child}".format(
        #     path=f'_{inflection.underscore(node_data.path.replace(".", "_"))}_'.replace(
        #         "__", "_"
        #     ),
        #     child=inflection.underscore(child.name),
        # )
        # global_functions += "\n    to{path}{child} = _to_{child}".format(
        #     path=f'_{inflection.underscore(node_data.path.replace(".", "_"))}_'.replace(
        #         "__", "_"
        #     ),
        #     child=inflection.underscore(child.name),
        # )

    base_function = """
def to_{path}(internal_{target_name}):
    return {return_class}({child_convert_logic} {member_convert_logic})
    
""".format(
        target_name=inflection.underscore(node_data.name),
        # nested_converters=nested_converters_string,
        path=f"{inflection.underscore(settings_type).replace('.', '_')}_{inflection.underscore(node_data.path).replace('.', '_')}"
        if node_data.path
        else f"{inflection.underscore(settings_type).replace('.', '_')}",
        return_class=return_class,
        member_convert_logic=member_convert_logic,
        child_convert_logic=child_convert_logic,
        # global_functions=global_functions,
    )
    indented_lines = list()
    for line in base_function.split("\n"):
        print(line)
        indented_lines.append("    " + line)
    return nested_converters_string + "\n" + "\n".join(indented_lines)


def create_to_internal_converter(node_data, settings_type: str):
    temp_internal_name = "internal_{name}".format(
        name=inflection.underscore(node_data.name)
    )
    nested_converters = [
        create_to_internal_converter(element, settings_type=settings_type)
        for element in node_data.children
    ]
    nested_converters_string = "\n".join(nested_converters)
    convert_member_logic = ""
    if node_data.member_variables:
        for member in node_data.member_variables:
            # convert_member_logic += "\n    if {name}.{member} is not None:\n".format(
            #     name=inflection.underscore(node_data.name), member=member.snake_case,
            # )
            convert_member_logic += "\n    {temp_internal_name}.{member} = _zivid.{settings_type}{path}".format(
                temp_internal_name=temp_internal_name,
                member=member.snake_case,
                path=".{path}.{member_as_class}({name}.{member})".format(
                    path=node_data.path,
                    member_as_class=member.camel_case,
                    name=inflection.underscore(node_data.name),
                    member=member.snake_case,
                )
                if node_data.path
                else ".{member_as_class}({name}.{member})".format(
                    member_as_class=member.camel_case,
                    name=inflection.underscore(node_data.name),
                    member=member.snake_case,
                ),
                settings_type=settings_type,
            )
            # convert_member_logic += "\n    else:"
            # convert_member_logic += "\n        {temp_internal_name}.{member} = _zivid.{settings_type}{path}".format(
            #     temp_internal_name=temp_internal_name,
            #     member=member.snake_case,
            #     path=".{path}.{member_as_class}()".format(
            #         path=node_data.path, member_as_class=member.camel_case,
            #     )
            #     if node_data.path
            #     else "()",
            #     settings_type=settings_type,
            # )

    convert_children_logic = ""
    # global_functions = ""
    if node_data.children:
        for child in node_data.children:
            convert_children_logic += "\n    {temp_internal_name}.{child_name} = to_internal_{child}({name}.{child_name})".format(
                temp_internal_name=temp_internal_name,
                child_name=inflection.underscore(child.name),
                child=f"{inflection.underscore(settings_type).replace('.', '_')}_{inflection.underscore(node_data.path).replace('.', '_')}_{inflection.underscore(child.name)}"
                if node_data.path
                else f"{inflection.underscore(settings_type).replace('.', '_')}_{inflection.underscore(child.name)}",
                name=inflection.underscore(node_data.name),
            )
            # expose internal_function through global
            # global_functions += "\n    global to_internal{path}{child}".format(
            #     path=f'_{inflection.underscore(node_data.path.replace(".", "_"))}_'.replace(
            #         "__", "_"
            #     ),
            #     child=inflection.underscore(child.name),
            # )
            # global_functions += "\n    to_internal{path}{child} = _to_internal_{child}".format(
            #     path=f'_{inflection.underscore(node_data.path.replace(".", "_"))}_'.replace(
            #         "__", "_"
            #     ),
            #     child=inflection.underscore(child.name),
            # )
    # else:
    #     convert_children_logic = "pass # no children"

    base_class = """
def to_internal_{path2}({target_name}):
    {temp_internal_name} = _zivid.{settings_type}{path}
    {convert_member_logic}
    {convert_children_logic}
    return {temp_internal_name}
""".format(
        target_name=inflection.underscore(node_data.name),
        # nested_converters=nested_converters_string,
        convert_member_logic=convert_member_logic,
        convert_children_logic=convert_children_logic,
        path="." + node_data.path + "()" if node_data.path else "()",
        path2=f"{inflection.underscore(settings_type).replace('.', '_')}_{inflection.underscore(node_data.path).replace('.', '_')}"
        if node_data.path
        else f"{inflection.underscore(settings_type).replace('.', '_')}",
        temp_internal_name=temp_internal_name,
        # global_functions=global_functions,
        settings_type=settings_type,
    )
    indented_lines = list()
    for line in base_class.splitlines():
        indented_lines.append("    " + line)
    return nested_converters_string + "\n" + "\n".join(indented_lines)


# import inspect
# from collections import namedtuple
# from dataclasses import dataclass
# from typing import Tuple
# import subprocess
# from _zivid.common import (
#     _create_class,
#     _imports,
#     _recursion,
# )


def common_class_generation(*, internal_class, settings_type, converter_import):
    data_model = _recursion(internal_class, indentation_level=0)
    with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
        temp_file = Path(temp_file.name)
        raw_text = _imports(
            internal=True,
            settings=True,
            additional_imports=(f"zivid.{converter_import}",),
        )
        raw_text += _create_class(data_model, settings_type=settings_type)

        new_lines = []
        for line in raw_text.splitlines():
            new_lines.append(line[4:])

        temp_file.write_text("\n".join(new_lines))
        print(temp_file.read_text())
        subprocess.check_output((f"black {temp_file}"), shell=True)
        print(temp_file.read_text())
        path_to_settings = (
            Path(__file__).resolve() / ".." / ".." / "zivid" / "settings__3d.py"
        ).resolve()
        path_to_settings.write_text(temp_file.read_text())


def common_to_internal_generation(*, internal_class, settings_type):
    data_model = _recursion(internal_class, indentation_level=0)
    with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
        temp_file = Path(temp_file.name)
        raw_text = _imports(internal=True, settings=False)
        raw_text += create_to_internal_converter(
            data_model, settings_type=settings_type
        )

        new_lines = []
        for line in raw_text.splitlines():
            new_lines.append(line[4:])

        temp_file.write_text("\n".join(new_lines))
        print(temp_file.read_text())
        subprocess.check_output((f"black {temp_file}"), shell=True)
        print(temp_file.read_text())
        path_to_settings = (
            Path(__file__).resolve() / ".." / ".." / "zivid" / "settings__3d.py"
        ).resolve()
        path_to_settings.write_text(temp_file.read_text())


def common_to_normal_generation(*, internal_class, settings_type):
    data_model = _recursion(internal_class, indentation_level=0)
    with tempfile.NamedTemporaryFile(suffix=".py") as temp_file:
        temp_file = Path(temp_file.name)
        raw_text = _imports(internal=False, settings=True)
        raw_text += create_to_not_internal_converter(
            data_model, settings_type=settings_type
        )

        new_lines = []
        for line in raw_text.splitlines():
            new_lines.append(line[4:])

        temp_file.write_text("\n".join(new_lines))
        print(temp_file.read_text())
        subprocess.check_output((f"black {temp_file}"), shell=True)
        print(temp_file.read_text())
        path_to_settings = (
            Path(__file__).resolve() / ".." / ".." / "zivid" / "settings__3d.py"
        ).resolve()
        path_to_settings.write_text(temp_file.read_text())
