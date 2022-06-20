
DEFAULT_SEEDS = [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]


def get_default_save_path(dataset_name, model_name, modifier=None):
    # Save directory
    save_dir = "models/{:s}/{:s}".format(dataset_name, dataset_name, model_name)

    # File name with optional additions
    file_name = "{:s}".format(model_name)
    if modifier is not None:
        file_name += "_" + str(modifier)
    file_name += ".pkl"

    # Full path
    path = "{:s}/{:s}".format(save_dir, file_name)

    # Return all
    return path, save_dir, file_name
