from abc import ABC

from pytorch_mil.data.benchmark.mnist.mnist_bags import FourMnistBagsDataset, FOURMNIST_N_CLASSES, \
    CountMnistBagsDataset, COUNTMNIST_N_CLASSES
from pytorch_mil.train.train_base import ClassificationTrainer, RegressionTrainer, NetTrainerMixin, GNNTrainerMixin


class FourMnistTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, FOURMNIST_N_CLASSES, model_clz, "models/mnist", model_params, train_params)

    def load_datasets(self, seed=None):
        return FourMnistBagsDataset.create_datasets(random_state=seed)

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
