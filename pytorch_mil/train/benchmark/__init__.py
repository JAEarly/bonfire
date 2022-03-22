from pytorch_mil.train.benchmark import crc_training, count_mnist_training, four_mnist_training


def get_trainer_clz(dataset_name, model_clz):
    if dataset_name == 'crc':
        trainer_clzs = crc_training.get_trainer_clzs()
    elif dataset_name == 'count_mnist':
        trainer_clzs = count_mnist_training.get_trainer_clzs()
    elif dataset_name == 'four_mnist':
        trainer_clzs = four_mnist_training.get_trainer_clzs()
    else:
        raise ValueError('No trainers found for dataset {:s}'.format(dataset_name))

    for trainer_clz in trainer_clzs:
        if model_clz.__base__ in trainer_clz.base_models:
            return trainer_clz

    raise ValueError('No trainer found for model {:} for dataset {:s}'.format(model_clz, dataset_name))
