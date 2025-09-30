import numpy as np
def list_of_dict_to_dict_of_list(list_of_dict: list[dict]) -> dict[np.ndarray]:
    dict_of_array = {}
    keys = list_of_dict[0].keys()
    for key in keys:
        if isinstance(list_of_dict[0][key], dict):
            dict_of_array[key] = list_of_dict_to_dict_of_list([d[key] for d in list_of_dict])
        else:
            dict_of_array[key] = [d[key] for d in list_of_dict]
    return dict_of_array