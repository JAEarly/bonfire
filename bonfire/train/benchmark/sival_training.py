from abc import ABC

from bonfire.data.benchmark.sival.sival_dataset import SivalDataset
from bonfire.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [SivalNetTrainer, SivalGNNTrainer]


class SivalTrainer(ClassificationTrainer, ABC):

    dataset_clz = SivalDataset

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class SivalNetTrainer(NetTrainerMixin, SivalTrainer):
    pass


class SivalGNNTrainer(GNNTrainerMixin, SivalTrainer):
    pass
