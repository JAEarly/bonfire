from abc import ABC

from bonfire.data.benchmark.dgr.dgr_dataset import DGRDataset
from bonfire.train.train_base import RegressionTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [DgrNetTrainer, DgrGNNTrainer]


class DgrTrainer(RegressionTrainer, ABC):

    dataset_clz = DGRDataset

    def __init__(self, device, model_clz, dataset_name, model_params=None, train_params_override=None):
        super().__init__(device, model_clz, dataset_name, dataset_params={}, model_params=model_params,
                         train_params_override=train_params_override)

    def load_datasets(self, seed=None):
        return DGRDataset.create_datasets(random_state=seed)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 2,
            'patience_interval': 4,
        }


class DgrNetTrainer(NetTrainerMixin, DgrTrainer):
    pass


class DgrGNNTrainer(GNNTrainerMixin, DgrTrainer):
    pass
