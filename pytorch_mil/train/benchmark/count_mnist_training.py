from abc import ABC

from pytorch_mil.data.benchmark.mnist.mnist_bags import CountMnistBagsDataset, COUNTMNIST_N_CLASSES
from pytorch_mil.train.train_base import RegressionTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [CountMnistNetTrainer, CountMnistGNNTrainer]


class CountMnistTrainer(RegressionTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, COUNTMNIST_N_CLASSES, model_clz, "models/count_mnist", model_params, train_params)

    def load_datasets(self, seed=None):
        return CountMnistBagsDataset.create_datasets(random_state=seed)

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
