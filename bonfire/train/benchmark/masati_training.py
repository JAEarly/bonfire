from abc import ABC

from pytorch_mil.data.benchmark.masati.masati_dataset import MasatiDataset
from pytorch_mil.train.train_base import RegressionTrainer, NetTrainerMixin, GNNTrainerMixin
from overrides import overrides
from torch import nn
from torch.utils.data import WeightedRandomSampler


def get_trainer_clzs():
    return [MasatiNetTrainer, MasatiGNNTrainer]


class MasatiTrainer(RegressionTrainer, ABC):

    dataset_clz = MasatiDataset

    def __init__(self, device, model_clz, dataset_name, model_params=None, train_params_override=None):
        super().__init__(device, model_clz, dataset_name, dataset_params={}, model_params=model_params,
                         train_params_override=train_params_override)

    def load_datasets(self, seed=None):
        return MasatiDataset.create_datasets(random_state=seed)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class MasatiNetTrainer(NetTrainerMixin, MasatiTrainer):
    pass


class MasatiGNNTrainer(GNNTrainerMixin, MasatiTrainer):
    pass
