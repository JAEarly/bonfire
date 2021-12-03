from .crc_training import CrcTrainer
from .mnist_training import MnistTrainer
from .musk_training import MuskTrainer
from .sival_training import SivalTrainer
from .tef_training import TefTrainer


def get_trainer_clz(dataset_name, model_name):
    if dataset_name == 'crc':
        return CrcTrainer.get_trainer_clz_from_model_name(model_name)
    if dataset_name == 'mnist':
        return MnistTrainer.get_trainer_clz_from_model_name(model_name)
    if dataset_name == 'musk':
        return MuskTrainer.get_trainer_clz_from_model_name(model_name)
    if dataset_name == 'sival':
        return SivalTrainer.get_trainer_clz_from_model_name(model_name)
    if dataset_name in ['tiger', 'elephant', 'fox']:
        return TefTrainer.get_trainer_clz_from_model_name(model_name)
    raise ValueError('No trainer found for dataset {:s}'.format(dataset_name))
