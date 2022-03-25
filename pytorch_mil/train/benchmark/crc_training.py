from abc import ABC

from pytorch_mil.data.benchmark.crc.crc_dataset import CrcDataset
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [CrcNetTrainer, CrcGNNTrainer]


class CrcTrainer(ClassificationTrainer, ABC):

    dataset_clz = CrcDataset

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class CrcNetTrainer(NetTrainerMixin, CrcTrainer):
    pass


class CrcGNNTrainer(GNNTrainerMixin, CrcTrainer):
    pass
