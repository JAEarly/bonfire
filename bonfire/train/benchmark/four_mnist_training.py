from abc import ABC

from pytorch_mil.data.benchmark.mnist.mnist_bags import FourMnistBagsDataset
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [FourMnistNetTrainer, FourMnistGNNTrainer]


class FourMnistTrainer(ClassificationTrainer, ABC):

    dataset_clz = FourMnistBagsDataset

    def get_default_train_params(self):
        return {
            'lr': 1e-5,
            'wd': 1e-3,
            'n_epochs': 100,
            'patience': 10,
        }


class FourMnistNetTrainer(NetTrainerMixin, FourMnistTrainer):
    pass


class FourMnistGNNTrainer(GNNTrainerMixin, FourMnistTrainer):
    pass
