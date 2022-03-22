from abc import ABC

from pytorch_mil.data.benchmark.crc.crc_dataset import CrcDataset, CRC_N_CLASSES
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


class CrcTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, CRC_N_CLASSES, model_clz, "models/crc", model_params, train_params)

    def load_datasets(self, seed=None):
        return CrcDataset.create_datasets(random_state=seed)

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
