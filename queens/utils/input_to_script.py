#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Convert input file to python script."""

import logging
import types
from pathlib import Path
from typing import Any

import black

from queens.distributions._distribution import Continuous
from queens.drivers._driver import Driver
from queens.iterators._iterator import Iterator
from queens.models.bmfmc import BMFMC
from queens.parameters.random_fields._random_field import RandomField
from queens.schedulers._scheduler import Scheduler
from queens.utils.from_config_create import VALID_TYPES, check_for_reference
from queens.utils.imports import get_module_attribute, get_option
from queens.utils.io import load_input_file

_logger = logging.getLogger(__name__)


DEFAULT_IMPORTS = [
    "from queens.global_settings import GlobalSettings",
    "from queens.main import run_iterator",
    "from queens.utils.io import load_result",
    "from queens.parameters.parameters import Parameters",
]
GLOBAL_SETTINGS_CONTEXT = [
    "with GlobalSettings(experiment_name=experiment_name,"
    " output_dir=output_dir, debug=False) as gs:"
]
RUN_ITERATOR = ["run_iterator(method, gs)"]
LOAD_RESULTS = [
    'result_file = gs.output_dir / f"{gs.experiment_name}.pickle"',
    "results = load_result(result_file)",
]
INDENT = " " * 4


class QueensPythonCode:
    """Class to create python script.

    Attributes:
        imports (list): list with the necessary imports
        run_iterator (list): list the run commands
        load_results (list): commands to load the results
        global_settings_context (list): commands to create the context
        code (list): list of code lines
        parameters (list): list of code lines for the parameters setup
        global_settings (list): list with all the global settings commands
        extern_imports (list): imports loaded with external python module
        create_main (bool): True if the script should contain a main function
    """

    def __init__(self) -> None:
        """Initialize object."""
        self.imports = DEFAULT_IMPORTS
        self.run_iterator = RUN_ITERATOR
        self.load_results = LOAD_RESULTS
        self.global_settings_context = GLOBAL_SETTINGS_CONTEXT
        self.code: list[str] = []
        self.parameters: list[str] = []
        self.global_settings: list[str] = []
        self.extern_imports: list[str] = []
        self.create_main = False

    def generate_script(self) -> str:
        """Format python code using black.

        Returns:
            formatted python code
        """
        mode = black.FileMode()
        fast = False
        return black.format_file_contents(self.generate_code(), fast=fast, mode=mode)

    def generate_code(self) -> str:
        """Generate the python code for the QUEENS run.

        Returns:
            python code
        """
        script = self.create_code_section(list(set(DEFAULT_IMPORTS)))
        if self.extern_imports:
            script += self.create_code_section(self.extern_imports, comment="External imports")

        indent = 0
        if self.create_main:
            # Due to dask processes, a main function is needed see
            # https://github.com/dask/distributed/issues/2520
            script += self.create_code_section(
                ["def run():"], comment="A main run is needed as you are using dask"
            )
            indent += 1

        script += self.create_code_section(
            self.global_settings, comment="Global settings", indent_level=indent
        )
        script += self.create_code_section(self.global_settings_context, indent_level=indent)

        indent += 1
        script += self.create_code_section(
            self.parameters, comment="Parameters", indent_level=indent
        )
        script += self.create_code_section(
            self.code, comment="Setup QUEENS stuff", indent_level=indent
        )
        script += self.create_code_section(
            self.run_iterator, comment="Actual analysis", indent_level=indent
        )
        script += self.create_code_section(
            self.load_results, comment="Load results", indent_level=indent
        )

        if self.create_main:
            script += self.create_code_section(
                ['if __name__=="__main__":', INDENT + "run()"], comment="main run"
            )
        return script

    @staticmethod
    def create_code_section(
        code_list: list[str], comment: str | None = None, indent_level: int = 0
    ) -> str:
        """Create python code section from a list of code lines.

        Args:
            code_list: list with code lines
            comment: comment for this code section
            indent_level: indent of this code block

        Returns:
            code section
        """
        section = "\n\n"
        indent = INDENT * indent_level
        if comment is not None:
            section += indent + "# " + comment
        section += f"\n{indent}"
        section += f"\n{indent}".join(code_list)
        section += "\n"
        return section


class VariableName:
    """Dummy class to differentiate between variable names and strings."""

    def __init__(self, name: str) -> None:
        """Initialize class.

        Args:
            name: variable name
        """
        self.name = name

    def __repr__(self):
        """Return repring str of object.

        Returns:
            return variable name
        """
        return self.name

    def __str__(self) -> str:
        """Return str of object.

        Returns:
            return variable name
        """
        return self.name


def dict_replace_infs(dictionary_to_modify: dict) -> dict:
    """Replace infs in nested dictionaries with `float(inf)`.

    The solution originates from https://stackoverflow.com/a/60776516
    Args:
        dictionary_to_modify: dictionary to modify

    Returns:
        modified dictionary
    """
    updated_dict = {}
    for key, value in dictionary_to_modify.items():
        if isinstance(value, dict):
            value = dict_replace_infs(value)
        elif isinstance(value, list):
            value = list_replace_infs(value)
        elif value in [float("inf"), float("-inf")]:
            updated_dict[key] = VariableName(f'float("{value}")')
        updated_dict[key] = value
    return updated_dict


def list_replace_infs(list_to_modify: list) -> list:
    """Replace infs in nested list with `float(inf)`.

    The solution originates from https://stackoverflow.com/a/60776516
    Args:
        list_to_modify: list to modify

    Returns:
        modified list
    """
    new_list = []
    for entry in list_to_modify:
        if isinstance(entry, list):
            entry = list_replace_infs(entry)
        elif isinstance(entry, dict):
            entry = dict_replace_infs(entry)
        elif entry in [float("inf"), float("-inf")]:
            entry = VariableName(f'float("{entry}")')
        new_list.append(entry)
    return new_list


def stringify(obj: Any) -> Any:
    """Wrap string in quotes for the source code.

    Args:
        obj: object for the code

    Returns:
        string version of the object
    """
    match obj:
        case str():
            return '"' + str(obj) + '"'
        case list():  # replace infs if necessary
            return list_replace_infs(obj)
        case dict():  # replace infs if necessary
            return dict_replace_infs(obj)
        # replace infs if necessary
        case _ if obj in [float("inf"), float("-inf")]:
            return VariableName(f'float("{obj}")')
        case _:
            return str(obj)


def create_initialization_call_from_class_and_arguments(
    class_name: str, arguments: dict[str, Any]
) -> str:
    """Create a initialization call.

    Args:
        class_name: name of the class to initialize
        arguments: keyword arguments for the object

    Returns:
        code  class_name(argument1=value1,...)
    """
    string_of_arguments = ", ".join([f"{k}={stringify(v)}" for (k, v) in arguments.items()])
    return f"{class_name}({string_of_arguments})"


def create_initialization_call(
    obj_description: dict[str, Any], python_code: QueensPythonCode
) -> str:
    """Create a initialization call.

    Args:
        obj_description: keyword arguments for the object
        python_code: object to store the code in

    Returns:
        code  "class_name(argument1=value1,...)"
    """
    object_class, class_name = get_module_class(obj_description, VALID_TYPES, python_code)

    if isinstance(object_class, types.FunctionType):
        return f"{class_name}"

    # add parameters
    if issubclass(object_class, (Iterator, Driver, BMFMC)):
        obj_description["parameters"] = VariableName("parameters")
    if issubclass(object_class, (Iterator, BMFMC)):
        obj_description["global_settings"] = VariableName("gs")

    if issubclass(object_class, Scheduler):
        obj_description["experiment_name"] = VariableName("gs.experiment_name")
        python_code.create_main = True

    return create_initialization_call_from_class_and_arguments(class_name, obj_description)


def assign_variable_value(variable_name: str, value: str) -> str:
    """Create code to assign value.

    Args:
        variable_name: name of the variable
        value: value to assign

    Returns:
        code line
    """
    return variable_name + "=" + value


def from_config_create_fields_code(
    random_field_preprocessor_options: dict, python_code: QueensPythonCode
):
    """Create code to preprocess random fields.

    Args:
        random_field_preprocessor_options: random field description
        python_code: object to store the code in
    """
    random_field_preprocessor = create_initialization_call(
        random_field_preprocessor_options, python_code
    )
    python_code.parameters.append(f"random_field_preprocessor = {random_field_preprocessor}")
    python_code.parameters.append("random_field_preprocessor.main_run()")
    python_code.parameters.append("random_field_preprocessor.write_random_fields_to_dat()")


def from_config_create_parameters(
    parameters_options: dict[str, Any], python_code: QueensPythonCode
) -> None:
    """Create a QUEENS parameter object from config.

    Args:
        parameters_options: Parameters description
        python_code: object to store the code in
    """
    joint_parameters_dict = {}
    for parameter_name, parameter_dict in parameters_options.items():
        parameter_class, distribution_class = get_module_class(
            parameter_dict, VALID_TYPES, python_code
        )
        if issubclass(parameter_class, Continuous):
            new_obj = create_initialization_call_from_class_and_arguments(
                distribution_class, parameter_dict
            )
        elif issubclass(parameter_class, RandomField):
            parameter_dict["coords"] = VariableName(
                f"random_field_preprocessor.coords_dict['{parameter_name}']"
            )
            new_obj = create_initialization_call_from_class_and_arguments(
                distribution_class, parameter_dict
            )
        else:
            raise NotImplementedError(f"Parameter type '{parameter_class.__name__}' not supported.")
        init_code = assign_variable_value(parameter_name, new_obj)
        python_code.parameters.append(init_code)
        parameter_object = VariableName(parameter_name)
        joint_parameters_dict[parameter_name] = parameter_object

    python_code.parameters.append(
        assign_variable_value(
            "parameters",
            create_initialization_call_from_class_and_arguments(
                "Parameters", joint_parameters_dict
            ),
        )
    )


def get_module_class(
    module_options: dict[str, Any],
    valid_types: dict[str, Any],
    code: QueensPythonCode,
    module_type_specifier: str = "type",
) -> tuple[type, str]:
    """Return module class defined in config file.

    Args:
        module_options: Module options
        valid_types: Dict of valid types with corresponding module paths and class names
        code: Object to store the code in
        module_type_specifier: Specifier for the module type

    Returns:
        Class from the module
        Name of the class
    """
    # determine which object to create
    module_type = module_options.pop(module_type_specifier)
    if module_options.get("external_python_module"):
        module_path = module_options.pop("external_python_module")
        module_class = get_module_attribute(module_path, module_type)
        code.imports.append("from queens.utils.import_utils import get_module_attribute")
        code.extern_imports.append(
            assign_variable_value(
                module_type, f'get_module_attribute("{module_path}", "{module_type}")'
            )
        )
        module_attribute = module_type
    else:
        module_class = get_option(valid_types, module_type)
        module_attribute = module_class.__name__
        module_path = module_class.__module__
        code.imports.append(f"from {module_path} import {module_attribute}")
    return module_class, module_attribute


def insert_new_obj(
    config: dict, new_obj_key: str, new_obj: str, python_code: QueensPythonCode
) -> dict[str, Any]:
    """Insert new object to the script.

    Note that this implementation deviates from the on in the fcc_utils

    Args:
        config: Description of queens run, or sub dictionary
        new_obj_key: Key of initialized object
        new_obj: Initialized object
        python_code: object to store the code in

    Returns:
        modified problem description
    """
    referenced_keys = []
    for key, value in config.items():
        if isinstance(value, dict):
            config[key] = insert_new_obj(value, new_obj_key, new_obj, python_code)
        elif key.endswith("_name") and value == new_obj_key:
            referenced_keys.append(key)

    for key in referenced_keys:
        config.pop(key)  # remove key "<example>_name"
        python_code.code.append(assign_variable_value(key.removesuffix("_name"), new_obj))
        config[key.removesuffix("_name")] = VariableName(key.removesuffix("_name"))
    return config


def from_config_create_script(
    config: dict,
    output_dir: Path,
) -> str:
    """Create a python script from input file.

    Args:
        config: Description of the QUEENS run
        output_dir: output directory

    Returns:
        python script for QUEENS
    """
    python_code = QueensPythonCode()

    experiment_name = config.pop("experiment_name")
    python_code.global_settings.append(
        assign_variable_value("experiment_name", '"' + experiment_name + '"')
    )
    python_code.global_settings.append(
        assign_variable_value("output_dir", '"' + str(output_dir) + '"')
    )

    random_field_preprocessor_options = config.pop("random_field_preprocessor", None)
    if random_field_preprocessor_options:
        from_config_create_fields_code(random_field_preprocessor_options, python_code)

    from_config_create_parameters(config.pop("parameters", {}), python_code)

    obj_key = None
    for _ in range(1000):  # Instead of 'while True' we only allow 1000 iterations for safety
        deadlock = True
        for obj_key, obj_dict in config.items():
            if isinstance(obj_dict, dict):
                reference_to_uninitialized_object = check_for_reference(obj_dict)
                if not reference_to_uninitialized_object:
                    deadlock = False
                    break
        if deadlock or obj_key is None:
            raise RuntimeError(
                "Queens run can not be configured due to missing 'method' "
                "description, circular dependencies or missing object descriptions! "
                f"Remaining uninitialized objects are: {list(config.keys())}"
            )

        obj_description = config.pop(obj_key)
        new_obj = create_initialization_call(obj_description, python_code)
        if obj_key == "method":
            if config:
                _logger.warning("Unused settings:")
                _logger.warning(config)

            python_code.code.append(assign_variable_value(obj_key, new_obj))
            return python_code.generate_script()

        config = insert_new_obj(config, obj_key, new_obj, python_code)
    raise RuntimeError()


def create_script_from_input_file(
    input_file: Path, output_dir: Path, script_path: Path | None = None
) -> None:
    """Create script from input file.

    Keep in mind that this does not work with jinja2 templates, only with

    Args:
        input_file : Input file path
        output_dir : Path to write the QUEENS run results
        script_path: Path for the python script
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    if script_path is None:
        script_path = input_file.with_suffix(".py")
    script_path = Path(script_path)

    _logger.info("Loading input file")
    config = load_input_file(input_file)

    _logger.info("Generating code")
    script = from_config_create_script(config, output_dir)

    _logger.info("Creating script")
    script_path.write_text(script, encoding="utf-8")
