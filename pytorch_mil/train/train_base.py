import copy
import inspect
import os
from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_mil.data.mil_graph_dataset import GraphDataloader
from pytorch_mil.model import models
from pytorch_mil.train import metrics, get_default_save_path
from pytorch_mil.train.metrics import ClassificationMetric, MaximizeRegressionMetric, MinimiseRegressionMetric


# -- UNUSED --
# def info_loss(outputs, targets, mi_scores):
#     prediction_loss = nn.CrossEntropyLoss()(outputs, targets.long())
#     mean_mi_scores = torch.mean(mi_scores, dim=0)
#     mi_global, mi_local, mi_prior = mean_mi_scores[0], mean_mi_scores[1], mean_mi_scores[2]
#     mi_loss = 0.8 * mi_global + 0.1 * mi_local + 0.1 * mi_prior
#     return prediction_loss, mi_loss, mean_mi_scores


def mil_collate_function(batch):
    # Collate function with variable size inputs
    #  Taken from https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2?u=ptrblck
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.as_tensor(target)
    return [data, target]


class Trainer(ABC):

    metric_clz = NotImplemented

    def __init__(self, device, model_clz, dataset_name,
                 dataset_params=None, model_params=None, train_params_override=None):
        self.device = device
        self.model_clz = model_clz
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.model_params = model_params
        self.train_params = self.get_default_train_params()
        self.update_train_params(train_params_override)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # All trainer concrete classes need to have a class attribute metric_type defined
        if cls.metric_clz is NotImplemented and not inspect.isabstract(cls):
            raise NotImplementedError('No metric_type defined for tuner {:}.'.format(cls))

    @classmethod
    @property
    @abstractmethod
    def dataset_clz(cls):
        pass

    @abstractmethod
    def get_default_train_params(self):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    @property
    def model_name(self):
        return self.model_clz.__name__

    @abstractmethod
    def plot_training(self, train_metrics, val_metrics):
        pass

    def load_datasets(self, seed=None):
        if self.dataset_params is not None:
            return self.dataset_clz.create_datasets(**self.dataset_params, seed=seed)
        return self.dataset_clz.create_datasets(seed=seed)

    def create_dataloader(self, dataset, batch_size):
        raise NotImplementedError("Base trainer class does not provide a create_dataloader implementation. "
                                  "You need to use a mixin: NetTrainerMixin, GNNTrainerMixin")

    def create_dataloaders(self, seed, batch_size=1):
        train_dataset, val_dataset, test_dataset = self.load_datasets(seed)
        train_dataloader = self.create_dataloader(train_dataset, batch_size)
        val_dataloader = self.create_dataloader(val_dataset, batch_size)
        test_dataloader = self.create_dataloader(test_dataset, batch_size)
        return train_dataloader, val_dataloader, test_dataloader

    def create_model(self):
        if self.model_params is not None:
            return self.model_clz(self.device, **self.model_params)
        return self.model_clz(self.device)

    def create_optimizer(self, model):
        lr = self.get_train_param('lr')
        weight_decay = self.get_train_param('weight_decay')
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)

    def get_train_param(self, key):
        return self.train_params[key]

    def update_train_params(self, new_params):
        if new_params is not None:
            for key, value in new_params.items():
                self.train_params[key] = value

    def train_epoch(self, model, optimizer, criterion, train_dataloader, val_dataloader):
        model.train()
        # epoch_train_loss = 0
        # epoch_prediction_loss = 0
        # epoch_mi_loss = 0
        # epoch_mi_sub_losses = torch.zeros(3)
        for data in train_dataloader:
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)
            # outputs, mi_scores = model(bags)
            loss = criterion(outputs, targets)
            # prediction_loss, mi_loss, mi_sub_losses = criterion(outputs, targets, mi_scores)
            # loss = prediction_loss + mi_loss
            loss.backward()
            optimizer.step()
            # epoch_train_loss += loss.item()
            # epoch_prediction_loss += prediction_loss.item()
            # epoch_mi_loss += mi_loss.item()
            # epoch_mi_sub_losses += mi_sub_losses.detach()
        # epoch_train_loss /= len(train_dataloader)
        # epoch_prediction_loss /= len(train_dataloader)

        # epoch_mi_loss /= len(train_dataloader)
        # epoch_mi_sub_losses /= len(train_dataloader)

        epoch_train_metrics, _ = metrics.eval_model(model, train_dataloader, criterion, self.metric_clz)
        epoch_val_metrics, _ = metrics.eval_model(model, val_dataloader, criterion, self.metric_clz)

        return epoch_train_metrics, epoch_val_metrics

    def train_model(self, model, train_dataloader, val_dataloader, trial=None):
        # Override current parameters with model suggested parameters
        self.update_train_params(model.suggest_train_params())

        model.to(self.device)
        model.train()

        optimizer = self.create_optimizer(model)
        criterion = self.get_criterion()

        early_stopped = False

        n_epochs = self.get_train_param('n_epochs')
        patience = self.get_train_param('patience')

        train_metrics = []
        val_metrics = []

        best_model = None

        if self.metric_clz.optimise_direction == 'maximize':
            best_key_metric = float("-inf")
        elif self.metric_clz.optimise_direction == 'minimise':
            best_key_metric = float("inf")
        else:
            raise ValueError('Invalid optimise direction {:}'.format(self.metric_clz.optimise_direction))

        with tqdm(total=n_epochs, desc='Training model', leave=False) as t:
            for epoch in range(n_epochs):
                # Train model for an epoch
                epoch_outputs = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloader)
                epoch_train_metrics, epoch_val_metrics = epoch_outputs

                # Early stopping
                if patience is not None:
                    new_key_metric = epoch_val_metrics.key_metric()
                    if self.metric_clz.optimise_direction == 'maximize' and new_key_metric > best_key_metric or \
                            self.metric_clz.optimise_direction == 'minimise' and new_key_metric < best_key_metric:
                        best_key_metric = new_key_metric
                        best_model = copy.deepcopy(model)
                        patience_tracker = 0
                    else:
                        patience_tracker += 1
                        if patience_tracker == patience:
                            early_stopped = True
                            break
                else:
                    best_model = copy.deepcopy(model)

                # Update progress bar
                train_metrics.append(epoch_train_metrics)
                val_metrics.append(epoch_val_metrics)
                t.set_postfix(train_metrics=epoch_train_metrics.short_string_repr(),
                              val_metrics=epoch_val_metrics.short_string_repr())
                t.update()

                # Update Optuna
                if trial is not None:
                    trial.report(epoch_val_metrics.key_metric(), epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        return best_model, train_metrics, val_metrics, early_stopped

    def train_single(self, seed=5, save_model=True, show_plot=True, verbose=True, trial=None):
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(seed, batch_size=1)
        model = self.create_model()
        train_outputs = self.train_model(model, train_dataloader, val_dataloader, trial=trial)
        del model
        best_model, train_metrics, val_metrics, early_stopped = train_outputs
        if hasattr(best_model, 'flatten_parameters'):
            best_model.flatten_parameters()
        train_results, val_results, test_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader,
                                                                         test_dataloader, self.get_criterion(),
                                                                         self.metric_clz, verbose=verbose)

        if save_model:
            path, save_dir, _ = get_default_save_path(self.dataset_name, self.model_name)
            print('Saving model to {:s}'.format(path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model.state_dict(), path)

        if show_plot:
            self.plot_training(train_metrics, val_metrics)

        return best_model, train_results, val_results, test_results, early_stopped

    def train_multiple(self, n_repeats=10, seeds=None):
        if seeds is not None and len(seeds) != n_repeats:
            print('Provided incorrect number of seeds: {:d} given but expected {:d}'.format(len(seeds), n_repeats))

        # Train multiple models
        results = []
        for i in range(n_repeats):
            print('Repeat {:d}/{:d}'.format(i + 1, n_repeats))
            repeat_seed = np.random.randint(low=1, high=1000) if seeds is None else seeds[i]
            print('Seed: {:d}'.format(repeat_seed))
            train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(repeat_seed, batch_size=1)
            model = self.create_model()
            best_model, _, _, _ = self.train_model(model, train_dataloader, val_dataloader)
            del model
            final_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader, test_dataloader,
                                                  self.get_criterion(), self.metric_clz, verbose=False)
            results.append(final_results)

            # Save model
            path, save_dir, _ = get_default_save_path(self.dataset_name, self.model_name, repeat=i)
            print('Saving model to {:s}'.format(path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model.state_dict(), path)

        metrics.output_results(results)


class ClassificationTrainer(Trainer, ABC):

    metric_clz = ClassificationMetric

    def get_criterion(self):
        return lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets.long())

    def plot_training(self, train_metrics, val_metrics):
        x_range = range(len(train_metrics))
        train_accs = [m.accuracy for m in train_metrics]
        train_losses = [m.loss for m in train_metrics]
        val_accs = [m.accuracy for m in val_metrics]
        val_losses = [m.loss for m in val_metrics]

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
        axes[0].plot(x_range, train_accs, label='Train')
        axes[0].plot(x_range, val_accs, label='Validation')
        axes[0].set_xlim(0, len(x_range))
        axes[0].set_ylim(min(min(train_accs), min(val_accs)) * 0.95, max(max(train_accs), max(val_accs)) * 1.05)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[1].plot(x_range, train_losses, label='Train')
        axes[1].plot(x_range, val_losses, label='Validation')
        axes[1].set_xlim(0, len(x_range))
        axes[1].set_ylim(min(min(train_losses), min(val_losses)) * 0.95, max(max(train_losses), max(val_losses)) * 1.05)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Cross Entropy Loss')
        axes[1].legend(loc='best')
        plt.show()


class MaximizeRegressionTrainer(Trainer, ABC):

    metric_clz = MaximizeRegressionMetric

    def get_criterion(self):
        raise NotImplementedError


class MinimiseRegressionTrainer(Trainer, ABC):

    metric_clz = MinimiseRegressionMetric

    def get_criterion(self):
        return lambda outputs, targets: nn.MSELoss()(outputs.squeeze(), targets.squeeze())

    def plot_training(self, train_metrics, val_metrics):
        x_range = range(len(train_metrics))
        train_losses = [m.loss for m in train_metrics]
        val_losses = [m.loss for m in val_metrics]

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        axis.plot(x_range, train_losses, label='Train')
        axis.plot(x_range, val_losses, label='Validation')
        axis.set_xlim(0, len(x_range))
        axis.set_ylim(min(min(train_losses), min(val_losses)) * 0.95, max(max(train_losses), max(val_losses)) * 1.05)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('MSE Loss')
        axis.legend(loc='best')
        plt.show()


class NetTrainerMixin:

    base_models = [models.InstanceSpaceNN, models.EmbeddingSpaceNN, models.AttentionNN, models.MiLstm]

    def create_dataloader(self, dataset, batch_size):
        if self.model_clz.__base__ not in self.base_models:
            raise ValueError('Invalid class {:} for trainer {:}.'.format(self.model_clz, self.__class__))
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)


class GNNTrainerMixin:

    base_models = [models.ClusterGNN]

    def create_dataloader(self, dataset, batch_size):
        if self.model_clz.__base__ not in self.base_models:
            raise ValueError('Invalid class {:} for trainer {:}.'.format(self.model_clz, self.__class__))
        return GraphDataloader(dataset)
