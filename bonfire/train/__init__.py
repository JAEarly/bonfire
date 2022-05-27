DEFAULT_SEEDS = [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]


def get_default_save_path(dataset_name, model_name, param_save_string="", repeat=None):
    # Save directory
    save_dir = "models/{:s}/{:s}_{:s}".format(dataset_name, dataset_name, model_name)

    # File name with optional additions
    file_name = "{:s}_{:s}".format(dataset_name, model_name)
    if param_save_string != "":
        file_name += "_" + param_save_string
    if repeat is not None:
        file_name += "_" + str(repeat)
    file_name += ".pkl"

    # Full path
    path = "{:s}/{:s}".format(save_dir, file_name)

    # Return all
    return path, save_dir, file_name
