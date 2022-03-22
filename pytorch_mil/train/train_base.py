import copy
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
from pytorch_mil.model.models import ClusterGNN
from pytorch_mil.train import metrics


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

    def __init__(self, device, n_classes, model_clz, save_dir, model_params=None, train_params_override=None):
        self.device = device
        self.n_classes = n_classes
        self.save_dir = save_dir
        self.model_clz = model_clz
        self.model_params = model_params
        self.train_params = self.get_default_train_params()
        self.update_train_params(train_params_override)

    @abstractmethod
    def load_datasets(self, seed=None):
        pass

    @abstractmethod
    def get_default_train_params(self):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    def create_dataloader(self, dataset, batch_size):
        raise NotImplementedError("Base trainer class does not provide a create_dataloader implementation. "
                                  "You need to use a mixin: NetTrainerMixin, GNNTrainerMixin")

    def create_dataloaders(self, seed, batch_size):
        train_dataset, val_dataset, test_dataset = self.load_datasets(seed)
        train_dataloader = self.create_dataloader(train_dataset, batch_size)
        val_dataloader = self.create_dataloader(val_dataset, batch_size)
        test_dataloader = self.create_dataloader(test_dataset, batch_size)
        return train_dataloader, val_dataloader, test_dataloader

    def create_model(self):
        if self.model_params is not None:
            return self.model_clz(self.device, **self.model_params)
        return self.model_clz(self.device)

    def get_model_name(self):
        return self.model_clz.__name__

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
            epoch_train_loss += loss.item()
            # epoch_prediction_loss += prediction_loss.item()
            # epoch_mi_loss += mi_loss.item()
            # epoch_mi_sub_losses += mi_sub_losses.detach()
        epoch_train_loss /= len(train_dataloader)
        # epoch_prediction_loss /= len(train_dataloader)

        # epoch_mi_loss /= len(train_dataloader)
        # epoch_mi_sub_losses /= len(train_dataloader)

        epoch_val_metrics = metrics.eval_model(model, val_dataloader, criterion, verbose=False)

        return epoch_train_loss, epoch_val_metrics

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

        train_losses = []
        val_metrics = []

        best_model = None
        best_val_loss = float("inf")

        with tqdm(total=n_epochs, desc='Training model', leave=False) as t:
            for epoch in range(n_epochs):
                # Train model for an epoch
                epoch_outputs = self.train_epoch(model, optimizer, criterion, train_dataloader, val_dataloader)
                epoch_train_loss, epoch_val_metrics = epoch_outputs

                # Early stopping
                if patience is not None:
                    if epoch_val_metrics.loss < best_val_loss:
                        best_val_loss = epoch_val_metrics.loss
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
                train_losses.append(epoch_train_loss)
                val_metrics.append(epoch_val_metrics)
                t.set_postfix(train_loss=epoch_train_loss, val_loss=epoch_val_metrics.loss)
                t.update()

                # Update Optuna
                if trial is not None:
                    trial.report(epoch_val_metrics.key_metric(), epoch)

                    # Handle pruning based on the intermediate value.
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

        return best_model, train_losses, val_metrics, early_stopped

    def train_single(self, seed=5, save_model=True, show_plot=True, verbose=True, trial=None):
        train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(seed, batch_size=1)
        model = self.create_model()
        train_outputs = self.train_model(model, train_dataloader, val_dataloader, trial=trial)
        del model
        best_model, train_losses, val_metrics, early_stopped = train_outputs
        if hasattr(best_model, 'flatten_parameters'):
            best_model.flatten_parameters()
        train_results, val_results, test_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader,
                                                                         test_dataloader, self.get_criterion(),
                                                                         verbose=verbose)

        if save_model:
            save_path = '{:s}/{:s}.pkl'.format(self.save_dir, self.get_model_name())
            print('Saving model to {:s}'.format(save_path))
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            torch.save(best_model.state_dict(), save_path)

        if show_plot:
            self.plot_training(train_losses, val_metrics)

        return best_model, train_results, val_results, test_results, early_stopped

    def train_multiple(self, num_repeats=10, seed=5):
        print('Training model with repeats')
        np.random.seed(seed=seed)

        model_save_dir = '{:s}/{:s}'.format(self.save_dir, self.get_model_name())
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # Train multiple models
        results = []
        for i in range(num_repeats):
            print('Repeat {:d}/{:d}'.format(i + 1, num_repeats))
            repeat_seed = np.random.randint(low=1, high=1000)
            print('Seed: {:d}'.format(repeat_seed))
            train_dataloader, val_dataloader, test_dataloader = self.create_dataloaders(repeat_seed, batch_size=1)
            model = self.create_model()
            best_model, _, _, _ = self.train_model(model, train_dataloader, val_dataloader)
            del model
            final_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader, test_dataloader,
                                                  self.get_criterion(), verbose=False)
            results.append(final_results)
            torch.save(best_model.state_dict(), '{:s}/{:s}_{:d}.pkl'.format(model_save_dir, self.get_model_name(), i))
        metrics.output_results(results)

    def plot_training(self, train_losses, val_metrics):
        fig, axes = plt.subplots(nrows=1, ncols=2)
        x_range = range(len(train_losses))
        axes[0].plot(x_range, train_losses, label='Train')
        axes[0].plot(x_range, [m[1] for m in val_metrics], label='Validation')
        axes[0].set_xlim(0, len(x_range))
        axes[0].set_ylim(min(min(train_losses), min([m[1] for m in val_metrics])) * 0.95,
                         max(max(train_losses), max([m[1] for m in val_metrics])) * 1.05)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(x_range, [m[0] for m in val_metrics], label='Validation')
        axes[1].set_xlim(0, len(x_range))
        axes[1].set_ylim(min([m[0] for m in val_metrics]) * 0.95, max([m[0] for m in val_metrics]) * 1.05)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend(loc='best')
        plt.show()


class ClassificationTrainer(Trainer, ABC):

    def get_criterion(self):
        return lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets.long())


class RegressionTrainer(Trainer, ABC):

    def get_criterion(self):
        return lambda outputs, targets: nn.MSELoss()(outputs.squeeze(), targets.squeeze())


class NetTrainerMixin:

    def create_dataloader(self, dataset, batch_size):
        assert ClusterGNN not in self.model_clz.__bases__
        return DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)


class GNNTrainerMixin:

    def create_dataloader(self, dataset, batch_size):
        assert ClusterGNN in self.model_clz.__bases__
        return GraphDataloader(dataset)
