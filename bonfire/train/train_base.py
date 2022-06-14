import copy
import inspect
import os
from abc import ABC, abstractmethod

import numpy as np
import optuna
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from bonfire.data.mil_graph_dataset import GraphDataloader
from bonfire.model import models
from bonfire.train import metrics, get_default_save_path
from bonfire.train.metrics import ClassificationMetric, RegressionMetric, CountRegressionMetric


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
            return self.dataset_clz.create_datasets(seed=seed, **self.dataset_params)
        return self.dataset_clz.create_datasets(seed=seed)

    def create_dataloader(self, dataset, shuffle, n_workers):
        raise NotImplementedError("Base trainer class does not provide a create_train_dataloader implementation. "
                                  "You need to use a mixin: NetTrainerMixin, GNNTrainerMixin")

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
        epoch_train_loss = 0
        # epoch_prediction_loss = 0
        # epoch_mi_loss = 0
        # epoch_mi_sub_losses = torch.zeros(3)
        for data in tqdm(train_dataloader, desc='Epoch Progress', leave=False):
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)
            # outputs, mi_scores = model(bags)
            loss = criterion(outputs, targets)
            # prediction_loss, mi_loss, mi_sub_losses = criterion(outputs, targets, mi_scores)
            # loss = prediction_loss + mi_loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            # epoch_prediction_loss += prediction_loss.item()
            # epoch_mi_loss += mi_loss.item()
            # epoch_mi_sub_losses += mi_sub_losses.detach()

        epoch_train_loss /= len(train_dataloader)
        # epoch_prediction_loss /= len(train_dataloader)

        # epoch_mi_loss /= len(train_dataloader)
        # epoch_mi_sub_losses /= len(train_dataloader)

        epoch_train_metrics = self.metric_clz.from_train_loss(epoch_train_loss)
        # epoch_train_metrics = metrics.eval_model(model, train_dataloader.dataset, criterion, self.metric_clz)
        epoch_val_metrics = None
        if val_dataloader is not None:
            epoch_val_metrics = metrics.eval_model(model, val_dataloader, self.metric_clz)

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
        patience_interval = self.get_train_param('patience_interval')

        train_metrics = []
        val_metrics = []

        best_model = None
        patience_tracker = 0

        if self.metric_clz.optimise_direction == 'maximize':
            best_key_metric = float("-inf")
        elif self.metric_clz.optimise_direction == 'minimize':
            best_key_metric = float("inf")
        else:
            raise ValueError('Invalid optimise direction {:}'.format(self.metric_clz.optimise_direction))

        print('Starting model training')
        if patience is not None:
            print('  Using patience of {:d} with interval {:d} (num patience epochs = {:d})'
                  .format(patience, patience_interval, patience * patience_interval))
        for epoch in range(n_epochs):
            print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
            # Train model for an epoch
            epoch_outputs = self.train_epoch(model, optimizer, criterion, train_dataloader,
                                             val_dataloader if epoch % patience_interval == 0 else None)
            epoch_train_metrics, epoch_val_metrics = epoch_outputs

            # Early stopping
            if patience is not None and epoch_val_metrics is not None:
                new_key_metric = epoch_val_metrics.key_metric()
                if self.metric_clz.optimise_direction == 'maximize' and new_key_metric > best_key_metric or \
                        self.metric_clz.optimise_direction == 'minimize' and new_key_metric < best_key_metric:
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

            # Update progress
            train_metrics.append(epoch_train_metrics)
            val_metrics.append(epoch_val_metrics)
            print(' Train: {:s}'.format(epoch_train_metrics.short_string_repr()))
            print('   Val: {:s}'.format(epoch_val_metrics.short_string_repr() if epoch_val_metrics else 'None'))

            # Update Optuna
            if trial is not None and epoch_val_metrics is not None:
                trial.report(epoch_val_metrics.key_metric(), epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return best_model, train_metrics, val_metrics, early_stopped

    def train_single(self, seed=5, save_model=True, show_plot=True, verbose=True, trial=None):
        train_dataset, val_dataset, test_dataset = self.load_datasets(seed)
        train_dataloader = self.create_dataloader(train_dataset, True, 2)
        val_dataloader = self.create_dataloader(val_dataset, False, 2)
        test_dataloader = self.create_dataloader(test_dataset, False, 2)
        model = self.create_model()
        train_outputs = self.train_model(model, train_dataloader, val_dataloader, trial=trial)
        del model
        best_model, train_metrics, val_metrics, early_stopped = train_outputs
        if hasattr(best_model, 'flatten_parameters'):
            best_model.flatten_parameters()
        train_results, val_results, test_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader,
                                                                         test_dataloader, self.metric_clz,
                                                                         verbose=verbose)
        if save_model:
            path, save_dir = self.get_model_save_path(best_model, None)
            if verbose:
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
            train_dataset, val_dataset, test_dataset = self.load_datasets(seed=repeat_seed)
            train_dataloader = self.create_train_dataloader(train_dataset, batch_size=1)
            model = self.create_model()
            best_model, _, _, _ = self.train_model(model, train_dataloader, val_dataset)
            del model

            if hasattr(best_model, 'flatten_parameters'):
                best_model.flatten_parameters()
            final_results = metrics.eval_complete(best_model, train_dataset, val_dataset, test_dataset,
                                                  self.metric_clz, verbose=False)
            results.append(final_results)

            # Save model
            path, save_dir = self.get_model_save_path(best_model, i)
            print('Saving model to {:s}'.format(path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model.state_dict(), path)

        metrics.output_results(results)

    def get_model_save_path(self, model, repeat):
        path, save_dir, _ = get_default_save_path(self.dataset_name, self.model_name,
                                                  param_save_string=model.get_param_save_string(), repeat=repeat)
        return path, save_dir


class ClassificationTrainer(Trainer, ABC):

    metric_clz = ClassificationMetric

    def get_criterion(self):
        return ClassificationMetric.criterion()

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


class RegressionTrainer(Trainer, ABC):

    metric_clz = RegressionMetric

    def get_criterion(self):
        return RegressionMetric.criterion()

    def plot_training(self, train_metrics, val_metrics):
        x_train_range = range(len(train_metrics))
        train_losses = [m.mse_loss for m in train_metrics]
        x_val_range = [idx for idx in range(len(val_metrics)) if val_metrics[idx] is not None]
        val_losses = [val_metrics[idx].mse_loss for idx in x_val_range]

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        axis.plot(x_train_range, train_losses, label='Train')
        axis.plot(x_val_range, val_losses, label='Validation')
        axis.set_xlim(0, len(x_train_range))
        axis.set_ylim(min(min(train_losses), min(val_losses)) * 0.95, max(max(train_losses), max(val_losses)) * 1.05)
        axis.set_xlabel('Epoch')
        axis.set_ylabel('MSE Loss')
        axis.legend(loc='best')
        plt.show()


class CountRegressionTrainer(RegressionTrainer, ABC):

    metric_clz = CountRegressionMetric

    def get_criterion(self):
        return CountRegressionMetric.criterion()


class NetTrainerMixin:

    base_models = [models.InstanceSpaceNN, models.EmbeddingSpaceNN, models.AttentionNN, models.MiLstm]

    def create_dataloader(self, dataset, shuffle, n_workers):
        if not any([base_clz in self.base_models for base_clz in inspect.getmro(self.model_clz)]):
            raise ValueError('Invalid class {:} for trainer {:}.'.format(self.model_clz, self.__class__))
        # TODO not using batch size
        return DataLoader(dataset, shuffle=shuffle, batch_size=1, num_workers=n_workers)


class GNNTrainerMixin:

    base_models = [models.ClusterGNN]

    def create_dataloader(self, dataset, shuffle, n_workers):
        if not any([base_clz in self.base_models for base_clz in inspect.getmro(self.model_clz)]):
            raise ValueError('Invalid class {:} for trainer {:}.'.format(self.model_clz, self.__class__))
        # TODO batch_size and n_workers for Graph data loader
        return GraphDataloader(dataset, shuffle)
