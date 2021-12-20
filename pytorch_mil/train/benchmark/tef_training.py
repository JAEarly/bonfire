from abc import ABC

from torch.utils.data import DataLoader

from pytorch_mil.data.tef_dataset import create_datasets, TEF_N_CLASSES
from pytorch_mil.model.models import ClusterGNN
from pytorch_mil.train.train_base import Trainer
from pytorch_mil.train.train_util import GraphDataloader


class TefTrainer(Trainer, ABC):

    def __init__(self, device, train_params, model_clz, dataset_name, model_params=None):
        super().__init__(device, TEF_N_CLASSES, model_clz, model_params, "models/{:s}".format(dataset_name),
                         train_params)
        self.dataset_name = dataset_name

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class TefNetTrainer(TefTrainer):

    def __init__(self, device, train_params, model_clz, dataset_name, model_params=None):
        super().__init__(device, train_params, model_clz, dataset_name, model_params=model_params)
        assert model_clz != ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(self.dataset_name, random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class TefGNNTrainer(TefTrainer):

    def __init__(self, device, train_params, model_clz, dataset_name, model_params=None):
        super().__init__(device, train_params, model_clz, dataset_name, model_params=model_params)
        assert model_clz == ClusterGNN

    def load_datasets(self, seed=None):
        train_dataset, val_dataset, test_dataset = create_datasets(self.dataset_name, random_state=seed)
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
