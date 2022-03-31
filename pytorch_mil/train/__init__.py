DEFAULT_SEEDS = [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]


def get_default_save_path(dataset_name, model_name, repeat=None):
    save_dir = "models/{:s}/{:s}_{:s}".format(dataset_name, dataset_name, model_name)
    file_name = "{:s}_{:s}_{:d}.pkl".format(dataset_name, model_name, repeat) if repeat is not None else \
                "{:s}_{:s}.pkl".format(dataset_name, model_name)
    path = "{:s}/{:s}".format(save_dir, file_name)
    return path, save_dir, file_name
