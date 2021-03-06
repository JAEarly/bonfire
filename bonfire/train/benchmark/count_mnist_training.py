from abc import ABC

from bonfire.data.benchmark.mnist.mnist_bags import CountMnistBagsDataset
from bonfire.train.train_base import RegressionTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [CountMnistNetTrainer, CountMnistGNNTrainer]


class CountMnistTrainer(RegressionTrainer, ABC):

    dataset_clz = CountMnistBagsDataset

    def get_default_train_params(self):
        return {
            'n_epochs': 100,
            'patience': 10,
            'lr': 1e-5,
            'weight_decay': 1e-3,
        }


class CountMnistNetTrainer(NetTrainerMixin, CountMnistTrainer):
    pass


class CountMnistGNNTrainer(GNNTrainerMixin, CountMnistTrainer):
    pass
