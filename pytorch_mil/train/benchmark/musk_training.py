from abc import ABC

from torch.utils.data import DataLoader

from pytorch_mil.data.musk_dataset import create_datasets, MUSK_N_CLASSES
from pytorch_mil.model.models import ClusterGNN
from pytorch_mil.train.train_base import Trainer
from pytorch_mil.data.mil_graph_dataset import GraphDataloader


class MuskTrainer(Trainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None, musk_two=False):
        save_dir = "models/musk2" if musk_two else "models/musk1"
        super().__init__(device, MUSK_N_CLASSES, model_clz, model_params, save_dir, train_params)
        self.musk_two = musk_two

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class MuskNetTrainer(MuskTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None, musk_two=False):
        super().__init__(device, train_params, model_clz, model_params=model_params, musk_two=musk_two)
        assert model_clz != ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(musk_two=self.musk_two, random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class MuskGNNTrainer(MuskTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None, musk_two=False):
        super().__init__(device, train_params, model_clz, model_params=model_params, musk_two=musk_two)
        assert model_clz == ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(musk_two=self.musk_two, random_state=seed)
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
