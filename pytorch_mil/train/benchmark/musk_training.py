from abc import ABC

from pytorch_mil.data.benchmark.musk.musk_dataset import MuskDataset, MUSK_N_CLASSES
from pytorch_mil.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


class MuskTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None, musk_two=False):
        save_dir = "models/musk2" if musk_two else "models/musk1"
        super().__init__(device, MUSK_N_CLASSES, model_clz, save_dir, model_params, train_params)
        self.musk_two = musk_two

    def load_datasets(self, seed=None):
        return MuskDataset.create_datasets(musk_two=self.musk_two, random_state=seed)

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'wd': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class MuskNetTrainer(NetTrainerMixin, MuskTrainer):
    pass


class MuskGNNTrainer(GNNTrainerMixin, MuskTrainer):
    pass
