import numpy as np

COMPRESS_TYPE = 'uncompressed array'


def convert_array_to_db_dict(numpy_array):
    """Convert numpy arrays in a dictionary format that can be understood by
    MongoDb.

    Args:
        numpy_array (np.array): array to compress

    Returns:
        dict: mongo db compatible dictionary
    """
    return {'ctype': COMPRESS_TYPE, 'shape': list(numpy_array.shape), 'value': numpy_array.tolist()}


def convert_db_dict_to_array(db_dict):
    """Convert a dictionary in a MongoDb compatible format to a numpy array.

    Args:
        db_dict (dict): Dict containing array

    Returns:
        (np.array) np.array
    """
    value = db_dict["value"]
    shape = db_dict["shape"]

    array = np.array(value)

    if list(array.shape) != shape:
        raise ValueError("Error while decompressing the array. ")

    return np.array(value)


def convert_nested_data_to_db_dict(u_container):
    """Restructure nested input data formats into dictionaries that are
    compatible with the MongoDB.

    Args:
        u_container (dict,list): list or dict with data to compress

    Returns:
        (dict,list): list or dict in to MongoDb compatible structure
    """
    if isinstance(u_container, dict):
        cdict = {}
        for key, value in u_container.items():

            # call method recursive in case another dict is encountered
            if isinstance(value, dict) or isinstance(value, list):
                cdict[key] = convert_nested_data_to_db_dict(value)

            # convert np.array to compatible dict
            else:
                if isinstance(value, np.ndarray):
                    cdict[key] = convert_array_to_db_dict(value)
                else:
                    cdict[key] = value

        return cdict

    elif isinstance(u_container, list):
        clist = []
        for value in u_container:
            if isinstance(value, dict) or isinstance(value, list):
                clist.append(convert_nested_data_to_db_dict(value))
            else:
                if isinstance(value, np.ndarray):
                    clist.append(convert_array_to_db_dict(value))
                else:
                    clist.append(value)

        return clist


def convert_nested_db_dicts_to_lists_or_arrays(db_data):
    """Restructure nested dictionaries in the MongoDb compatible format to
    return the original input format of either list or dict type.

    Args:
        db_data (dict,list): dict or list in MongoDb compatible format

    Returns:
        (dict,list): list or dict in original data format or structure
    """
    if isinstance(db_data, dict):
        if 'ctype' in db_data and db_data['ctype'] == COMPRESS_TYPE:
            try:
                return convert_db_dict_to_array(db_data)
            except:
                raise Exception('Container does not contain a valid array.')
        else:
            udict = {}
            for key, value in db_data.items():
                if isinstance(value, dict) or isinstance(value, list):
                    udict[key] = convert_nested_db_dicts_to_lists_or_arrays(value)
                else:
                    udict[key] = value

            return udict
    elif isinstance(db_data, list):
        ulist = []
        for value in db_data:
            if isinstance(value, dict) or isinstance(value, list):
                ulist.append(convert_nested_db_dicts_to_lists_or_arrays(value))
            else:
                ulist.append(value)

        return ulist