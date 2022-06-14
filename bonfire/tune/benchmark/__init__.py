from bonfire.tune.benchmark import count_mnist_tuning, crc_tuning, four_mnist_tuning, sival_tuning, dgr_tuning

tuner_dict = {}


def get_tuner_clz(model_clz):
    # Register tuners if first time
    if not tuner_dict:
        for clz in [count_mnist_tuning, crc_tuning, four_mnist_tuning, sival_tuning, dgr_tuning]:
            for tuner_clz in clz.get_tuner_clzs():
                tuner_dict[tuner_clz.model_clz] = tuner_clz
    # Retrieve tuner for this model
    if model_clz in tuner_dict:
        return tuner_dict[model_clz]
    raise NotImplementedError('No tuner implemented for class {:}'.format(model_clz))
