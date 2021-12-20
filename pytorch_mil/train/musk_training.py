from abc import ABC

from pytorch_mil.model.models import ClusterGNN
from torch.utils.data import DataLoader

from pytorch_mil.data.musk_dataset import create_datasets, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS
from pytorch_mil.train.train_base import Trainer
from pytorch_mil.train.train_util import GraphDataloader


class MuskTrainer(Trainer, ABC):

    def __init__(self, device, model_clz, model_yobj=None, musk_two=False):
        save_dir = "models/musk2" if musk_two else "models/musk1"
        model_config_path = "config/musk1_models.yaml"
        super().__init__(device, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, model_clz,
                         save_dir, model_config_path, model_yobj)
        self.musk_two = musk_two

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }

    @staticmethod
    def get_trainer_clz_from_model_name(model_name):
        return MuskGNNTrainer if model_name == 'ClusterGNN' else MuskNetTrainer


class MuskNetTrainer(MuskTrainer):

    def __init__(self, device, model_clz, model_yobj=None, musk_two=False):
        super().__init__(device, model_clz, model_yobj=model_yobj, musk_two=musk_two)
        assert model_clz != ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(musk_two=self.musk_two, random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class MuskGNNTrainer(MuskTrainer):

    def __init__(self, device, model_clz, model_yobj=None, musk_two=False):
        super().__init__(device, model_clz, model_yobj=model_yobj, musk_two=musk_two)
        assert model_clz == ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(musk_two=self.musk_two, random_state=seed)
        train_dataset.add_edges()
        val_dataset.add_edges()
        test_dataset.add_edges()
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
