from abc import ABC, abstractmethod

import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Metric(ABC):

    @abstractmethod
    def key_metric(self):
        """
        This is the metric that we're going to maximise/minimise when doing hyperparameter tuning
        :return:
        """

    @property
    @abstractmethod
    def optimise_direction(self):
        """
        The direction that we want to optimise when doing hyperparameter tuning
        maximise for classification (accuracy)
        minimise for regression (loss)
        """

    @staticmethod
    @abstractmethod
    def criterion():
        pass

    @staticmethod
    @abstractmethod
    def calculate_metric(probas, targets, labels):
        pass

    @staticmethod
    @abstractmethod
    def from_train_loss(train_loss):
        pass

    @abstractmethod
    def short_string_repr(self):
        pass

    @abstractmethod
    def out(self):
        pass


class ClassificationMetric(Metric):

    optimise_direction = 'maximize'

    def __init__(self, accuracy, loss, conf_mat):
        self.accuracy = accuracy
        self.loss = loss
        self.conf_mat = conf_mat

    def key_metric(self):
        return self.accuracy

    @staticmethod
    def criterion():
        return lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets.long())

    @staticmethod
    def calculate_metric(preds, targets, labels):
        _, probas = torch.max(F.softmax(preds, dim=1), dim=1)
        acc = accuracy_score(targets.long(), probas)
        loss = ClassificationMetric.criterion()(preds, targets).item()
        conf_mat = pd.DataFrame(
            confusion_matrix(targets.long(), probas, labels=labels),
            index=pd.Index(labels, name='Actual'),
            columns=pd.Index(labels, name='Predicted')
        )
        return ClassificationMetric(acc, loss, conf_mat)

    @staticmethod
    def from_train_loss(train_loss):
        return ClassificationMetric(None, train_loss, None)

    def short_string_repr(self):
        return "{{Acc: {:.3f}; Loss: {:.3f}}}".format(self.accuracy, self.loss)

    def out(self):
        print('Acc: {:.3f}'.format(self.accuracy))
        print('Loss: {:.3f}'.format(self.loss))
        print(self.conf_mat)


class RegressionMetric(Metric, ABC):

    optimise_direction = 'minimise'

    def __init__(self, mse_loss, mae_loss):
        self.mse_loss = mse_loss
        self.mae_loss = mae_loss

    def key_metric(self):
        return self.mse_loss

    @staticmethod
    def criterion():
        return lambda outputs, targets: nn.MSELoss()(outputs.squeeze(), targets.squeeze())

    @staticmethod
    def calculate_metric(preds, targets, labels):
        mse_loss = RegressionMetric.criterion()(preds, targets).item()
        mae_loss = nn.L1Loss()(preds.squeeze(), targets.squeeze()).item()
        return RegressionMetric(mse_loss, mae_loss)

    @staticmethod
    def from_train_loss(train_loss):
        return RegressionMetric(train_loss, None)

    def short_string_repr(self):
        return "{{MSE Loss: {:.3f}; ".format(self.mse_loss) + \
               ("MAE Loss: {:.3f}}}".format(self.mae_loss) if self.mae_loss is not None else "MAE Loss: None")

    def out(self):
        print('MSE Loss: {:.3f}'.format(self.mse_loss))
        print('MAE Loss: {:.3f}'.format(self.mae_loss))


def eval_complete(model, train_dataset, val_dataset, test_dataset, metric: Metric, verbose=False):
    train_results = eval_model(model, train_dataset, metric)
    if verbose:
        print('\n-- Train Results --')
        train_results.out()
    val_results = eval_model(model, val_dataset, metric)
    if verbose:
        print('\n-- Val Results --')
        val_results.out()
    test_results = eval_model(model, test_dataset, metric)
    if verbose:
        print('\n-- Test Results --')
        test_results.out()
    return train_results, val_results, test_results


def eval_model(model, dataset, metric: Metric, n_workers=2, batch_size=2):
    dataloader = DataLoader(dataset, shuffle=False, num_workers=n_workers, batch_size=batch_size)
    model.eval()
    with torch.no_grad():
        all_preds = []
        for data in tqdm(dataloader, desc='Evaluating', leave=False):
            bags = data[0]
            bag_pred = model(bags)
            all_preds.append(bag_pred.cpu())
        labels = list(range(model.n_classes))
        all_preds = torch.cat(all_preds)
        bag_metric = metric.calculate_metric(all_preds, dataset.targets, labels)
        return bag_metric


def output_results(model_names, results_arr):
    n_models, n_repeats, n_splits = results_arr.shape
    assert n_models == len(model_names)
    assert n_splits == 3

    results_type = type(results_arr[0][0][0])
    if results_type == ClassificationMetric:
        output_classification_results(model_names, results_arr)
    elif issubclass(results_type, RegressionMetric):
        output_regression_results(model_names, results_arr)
    else:
        raise NotImplementedError('No results output for metrics {:}'.format(results_type))


def output_classification_results(model_names, results_arr):
    n_models, n_repeats, _ = results_arr.shape
    results = np.empty((n_models, 6), dtype=object)
    for model_idx in range(n_models):
        model_results = results_arr[model_idx]
        expanded_model_results = np.empty((n_repeats, 6), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.accuracy, train_results.loss,
                                                     val_results.accuracy, val_results.loss,
                                                     test_results.accuracy, test_results.loss]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        for metric_idx in range(6):
            results[model_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    rows = [['Model Name', 'Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss']]
    for model_idx in range(n_models):
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))
    print('Done!')


def output_regression_results(model_names, results_arr):
    n_models, n_repeats, _ = results_arr.shape
    results = np.empty((n_models, 6), dtype=object)
    for model_idx in range(n_models):
        model_results = results_arr[model_idx]
        expanded_model_results = np.empty((n_repeats, 6), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.mse_loss, train_results.mae_loss,
                                                     val_results.mse_loss, val_results.mae_loss,
                                                     test_results.mse_loss, test_results.mae_loss]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        for metric_idx in range(6):
            results[model_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    rows = [['Model Name', 'Train MSE', 'Train MAE', 'Val MSE', 'Val MAE', 'Test MSE', 'Test MAE']]
    for model_idx in range(n_models):
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))
    print('Done!')
