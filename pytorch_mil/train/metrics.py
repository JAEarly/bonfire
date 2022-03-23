from abc import ABC, abstractmethod

import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from typing import List, Tuple


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
        minimise for regression (loss
        :return:
        """

    @staticmethod
    @abstractmethod
    def calculate_metric(probas, targets, criterion, labels):
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
    def calculate_metric(probas, targets, criterion, labels):
        _, preds = torch.max(F.softmax(probas, dim=1), dim=1)
        acc = accuracy_score(targets, preds)
        loss = criterion(probas, targets).item()
        conf_mat = pd.DataFrame(
            confusion_matrix(targets, preds, labels=labels),
            index=pd.Index(labels, name='Actual'),
            columns=pd.Index(labels, name='Predicted')
        )
        return ClassificationMetric(acc, loss, conf_mat)


class RegressionMetric(Metric, ABC):

    def __init__(self, loss):
        self.loss = loss

    def key_metric(self):
        return self.loss

    @staticmethod
    def calculate_metric(probas, targets, criterion, labels):
        loss = criterion(probas, targets).item()
        return RegressionMetric(loss)


class MaximizeRegressionMetric(RegressionMetric):
    optimise_direction = 'maximize'


class MinimiseRegressionMetric(RegressionMetric):
    optimise_direction = 'minimise'


def eval_complete(model, train_dataloader, val_dataloader, test_dataloader, criterion, metric: Metric, verbose=False):
    if verbose:
        print('\n-- Train Results --')
    train_results = eval_model(model, train_dataloader, criterion, metric)
    if verbose:
        print('\n-- Val Results --')
    val_results = eval_model(model, val_dataloader, criterion, metric)
    if verbose:
        print('\n-- Test Results --')
    test_results = eval_model(model, test_dataloader, criterion, metric)
    return train_results, val_results, test_results


def eval_model(model, dataloader, criterion, metric: Metric):
    model.eval()
    with torch.no_grad():
        labels = list(range(model.n_classes))
        all_probas = []
        all_targets = []
        for data in dataloader:
            bags, targets = data[0], data[1]
            bag_preds = model(bags)
            all_probas.append(bag_preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
        all_targets = torch.cat(all_targets).long()
        all_probas = torch.cat(all_probas)
        return metric.calculate_metric(all_probas, all_targets, criterion, labels)


def output_results(results: List[Tuple[Metric, Metric, Metric]]):
    results_type = type(results[0][0])
    if results_type == ClassificationMetric:
        output_classification_results(results)
    elif results_type == RegressionMetric:
        output_regression_results(results)
    else:
        raise NotImplementedError('No results output for metrics {:}'.format(results_type))


def output_classification_results(results: List[Tuple[ClassificationMetric, ClassificationMetric,
                                                      ClassificationMetric]]):
    raw_results = []
    for (train_results, val_results, test_results) in results:
        raw_results.append([train_results.accuracy, train_results.loss,
                            val_results.accuracy, val_results.loss,
                            test_results.accuracy, test_results.loss])
    results = np.asarray(results)
    rows = [['Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss']]
    results_row = []
    for i in range(6):
        values = results[:, i]
        mean = np.mean(values)
        sem = np.std(values) / np.sqrt(len(values))
        results_row.append('{:.4f} +- {:.4f}'.format(mean, sem))
    rows.append(results_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 6)
    table.set_cols_align(['c'] * 6)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))
    print('Done!')


def output_regression_results(results: List[Tuple[RegressionMetric, RegressionMetric, RegressionMetric]]):
    raw_results = []
    for (train_results, val_results, test_results) in results:
        raw_results.append([train_results.loss, val_results.loss, test_results.loss])
    results = np.asarray(results)
    rows = [['Train Loss', 'Val Loss', 'Test Loss']]
    results_row = []
    for i in range(3):
        values = results[:, i]
        mean = np.mean(values)
        sem = np.std(values) / np.sqrt(len(values))
        results_row.append('{:.4f} +- {:.4f}'.format(mean, sem))
    rows.append(results_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 3)
    table.set_cols_align(['c'] * 3)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))
    print('Done!')
