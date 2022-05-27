from bonfire.tune.benchmark import count_mnist_tuning, crc_tuning, four_mnist_tuning, sival_tuning

tuner_dict = {}


def get_tuner_clz(model_clz):
    # Register tuners if first time
    if not tuner_dict:
        for tuner_clz in count_mnist_tuning.get_tuner_clzs():
            tuner_dict[tuner_clz.model_clz] = tuner_clz
        for tuner_clz in crc_tuning.get_tuner_clzs():
            tuner_dict[tuner_clz.model_clz] = tuner_clz
        for tuner_clz in four_mnist_tuning.get_tuner_clzs():
            tuner_dict[tuner_clz.model_clz] = tuner_clz
        for tuner_clz in sival_tuning.get_tuner_clzs():
            tuner_dict[tuner_clz.model_clz] = tuner_clz
    # Retrieve tuner for this model
    if model_clz in tuner_dict:
        return tuner_dict[model_clz]
    raise NotImplementedError('No tuner implemented for class {:}'.format(model_clz))
