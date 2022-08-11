"""Data processor.

Extract data from simulation output.
"""
from pqueens.utils.import_utils import get_module_class


def from_config_create_data_processor(config, data_processor_name):
    """Create DataProcessor object from problem description.

    Args:
        config (dict): input json file with problem description
        data_processor_name (str): Name of the data processor

    Returns:
        data_processor (obj): data_processor object
    """
    valid_types = {
        'csv': ['.data_processor_csv_data', 'DataProcessorCsv'],
        'ensight': ['.data_processor_ensight', 'DataProcessorEnsight'],
        'ensight_interface_discrepancy': [
            '.data_processor_ensight_interface',
            'DataProcessorEnsightInterfaceDiscrepancy',
        ],
    }

    data_processor_options = config.get(data_processor_name)
    if not data_processor_options:
        raise ValueError(
            "The 'data processor' options were not found in the input file! "
            f"You specified the data processor name '{data_processor_name}'. Abort..."
        )

    data_processor_options = config[data_processor_name]
    data_processor_type = data_processor_options.get("type")
    data_processor_class = get_module_class(
        data_processor_options, valid_types, data_processor_type
    )

    data_processor = data_processor_class.from_config_create_data_processor(
        config, data_processor_name
    )

    return data_processor
