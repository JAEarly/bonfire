from abc import ABC

from pytorch_mil.data.benchmark.tef.tef_dataset import TefDataset, TEF_N_CLASSES
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


class TefTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, dataset_name, model_params=None):
        super().__init__(device, TEF_N_CLASSES, model_clz, "models/{:s}".format(dataset_name), model_params,
                         train_params)
        self.dataset_name = dataset_name

    def load_datasets(self, seed=None):
        return TefDataset.create_datasets(self.dataset_name, random_state=seed)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class TefNetTrainer(NetTrainerMixin, TefTrainer):
    pass


class TefGNNTrainer(GNNTrainerMixin, TefTrainer):
    pass
