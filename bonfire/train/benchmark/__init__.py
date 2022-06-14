from bonfire.train.benchmark import crc_training, count_mnist_training, dgr_training, four_mnist_training, \
    masati_training, musk_training, sival_training, tef_training


def get_trainer_clz(dataset_name, model_clz):
    if dataset_name == 'crc':
        trainer_clzs = crc_training.get_trainer_clzs()
    elif dataset_name == 'count_mnist':
        trainer_clzs = count_mnist_training.get_trainer_clzs()
    elif dataset_name == 'dgr':
        trainer_clzs = dgr_training.get_trainer_clzs()
    elif dataset_name == 'four_mnist':
        trainer_clzs = four_mnist_training.get_trainer_clzs()
    elif dataset_name == 'masati':
        trainer_clzs = masati_training.get_trainer_clzs()
    elif dataset_name == 'musk':
        trainer_clzs = musk_training.get_trainer_clzs()
    elif dataset_name == 'sival':
        trainer_clzs = sival_training.get_trainer_clzs()
    elif dataset_name in ['tiger', 'elephant', 'fox']:
        trainer_clzs = tef_training.get_trainer_clzs()
    else:
        raise ValueError('No trainers found for dataset {:s}'.format(dataset_name))

    for trainer_clz in trainer_clzs:
        if model_clz.__base__ in trainer_clz.base_models:
            return trainer_clz

    raise ValueError('No trainer found for model {:} for dataset {:s}'.format(model_clz, dataset_name))
