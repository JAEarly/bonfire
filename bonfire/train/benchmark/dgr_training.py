from abc import ABC

from pytorch_mil.data.benchmark.dgr.dgr_dataset import DGRDataset, DGR_N_CLASSES
from pytorch_mil.train.train_base import MinimiseRegressionTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [DgrNetTrainer, DgrGNNTrainer]


class DgrTrainer(MinimiseRegressionTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, "drg", DGR_N_CLASSES, model_clz, model_params, train_params)

    def load_datasets(self, seed=None):
        return DGRDataset.create_datasets(random_state=seed)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class DgrNetTrainer(NetTrainerMixin, DgrTrainer):
    pass


class DgrGNNTrainer(GNNTrainerMixin, DgrTrainer):
    pass
