from abc import ABC

from torch.utils.data import DataLoader

from pytorch_mil.data.mil_graph_dataset import GraphDataloader
from pytorch_mil.data.mnist_bags import FourMnistBagsDataset, FOURMNIST_N_CLASSES, \
    CountMnistBagsDataset, COUNTMNIST_N_CLASSES
from pytorch_mil.model.benchmark import mnist_models
from pytorch_mil.train.train_base import ClassificationTrainer, RegressionTrainer


class FourMnistTrainer(ClassificationTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, FOURMNIST_N_CLASSES, model_clz, model_params, "models/mnist", train_params)

    def get_default_train_params(self):
        return {
            'lr': 1e-5,
            'wd': 1e-3,
            'n_epochs': 100,
            'patience': 10,
        }


class FourMnistNetTrainer(FourMnistTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz != mnist_models.FourMnistGNN

    def load_dataloaders(self, seed=None):
        train_dataset, val_dataset, test_dataset = FourMnistBagsDataset.create_datasets(random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class FourMnistGNNTrainer(FourMnistTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz == mnist_models.FourMnistGNN

    def load_dataloaders(self, seed=None):
        train_dataset, val_dataset, test_dataset = FourMnistBagsDataset.create_datasets(random_state=seed)
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader


class CountMnistTrainer(RegressionTrainer, ABC):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, COUNTMNIST_N_CLASSES, "models/count_mnist", train_params, model_clz,
                         model_params=model_params)

    def get_default_train_params(self):
        return {
            'n_epochs': 100,
            'patience': 10,
            'lr': 1e-5,
            'weight_decay': 1e-3,
        }


class CountMnistNetTrainer(CountMnistTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz != mnist_models.CountMnistGNN

    def load_dataloaders(self, seed=None):
        train_dataset, val_dataset, test_dataset = CountMnistBagsDataset.create_datasets(random_state=seed)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
        return train_dataloader, val_dataloader, test_dataloader


class CountMnistGNNTrainer(CountMnistTrainer):

    def __init__(self, device, train_params, model_clz, model_params=None):
        super().__init__(device, train_params, model_clz, model_params=model_params)
        assert model_clz == mnist_models.CountMnistGNN

    def load_dataloaders(self, seed=None):
        train_dataset, val_dataset, test_dataset = CountMnistBagsDataset.create_datasets(random_state=seed)
        train_dataloader = GraphDataloader(train_dataset)
        val_dataloader = GraphDataloader(val_dataset)
        test_dataloader = GraphDataloader(test_dataset)
        return train_dataloader, val_dataloader, test_dataloader
