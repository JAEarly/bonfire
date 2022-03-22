from abc import ABC

from pytorch_mil.data.benchmark.sival.sival_dataset import SIVAL_N_CLASSES, SivalDataset
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


class SivalTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, SIVAL_N_CLASSES, model_clz, "models/sival", model_params, train_params)

    def load_datasets(self, seed=None):
        return SivalDataset.create_datasets(random_state=seed)

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
