from abc import ABC

from bonfire.data.benchmark.musk.musk_dataset import Musk1Dataset, Musk2Dataset
from bonfire.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [Musk1NetTrainer, Musk1GNNTrainer, Musk2NetTrainer, Musk2GNNTrainer]


class MuskTrainer(ClassificationTrainer, ABC):

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'wd': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class Musk1Trainer(MuskTrainer, ABC):
    dataset_clz = Musk1Dataset


class Musk2Trainer(MuskTrainer, ABC):
    dataset_clz = Musk2Dataset


class Musk1NetTrainer(NetTrainerMixin, Musk1Trainer):
    pass


class Musk1GNNTrainer(GNNTrainerMixin, Musk1Trainer):
    pass


class Musk2NetTrainer(NetTrainerMixin, Musk2Trainer):
    pass


class Musk2GNNTrainer(GNNTrainerMixin, Musk2Trainer):
    pass
