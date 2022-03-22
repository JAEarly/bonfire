from pytorch_mil.model.benchmark import crc_models, count_mnist_models


def get_model_clz(dataset_name, model_name):
    if dataset_name == 'crc':
        model_clzs = crc_models.get_model_clzs()
    elif dataset_name == 'count_mnist':
        model_clzs = count_mnist_models.get_model_clzs()
    else:
        raise ValueError('No models registered for dataset {:}'.format(dataset_name))

    for model_clz in model_clzs:
        if model_name == model_clz.__base__.__name__:
            return model_clz
    raise ValueError('No model with name {:s} found for dataset {:s}'.format(model_name, dataset_name))
