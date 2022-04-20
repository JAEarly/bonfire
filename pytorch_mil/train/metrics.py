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

    @abstractmethod
    def short_string_repr(self):
        pass

    @abstractmethod
    def output(self):
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
    def calculate_metric(preds, targets, criterion, labels):
        _, probas = torch.max(F.softmax(preds, dim=1), dim=1)
        acc = accuracy_score(targets.long(), probas)
        loss = criterion(preds, targets).item()
        conf_mat = pd.DataFrame(
            confusion_matrix(targets.long(), probas, labels=labels),
            index=pd.Index(labels, name='Actual'),
            columns=pd.Index(labels, name='Predicted')
        )
        return ClassificationMetric(acc, loss, conf_mat)

    def short_string_repr(self):
        return "{{Acc: {:.3f}; Loss: {:.3f}}}".format(self.accuracy, self.loss)

    def output(self):
        print('Acc: {:.3f}'.format(self.accuracy))
        print('Loss: {:.3f}'.format(self.loss))
        print(self.conf_mat)


class RegressionMetric(Metric, ABC):

    def __init__(self, loss):
        self.loss = loss

    def key_metric(self):
        return self.loss

    @classmethod
    def calculate_metric(cls, preds, targets, criterion, labels):
        loss = criterion(preds, targets).item()
        return cls(loss)

    def short_string_repr(self):
        return "{{Loss: {:.3f}}}".format(self.loss)

    def output(self):
        print('Loss: {:.3f}'.format(self.loss))


class MaximizeRegressionMetric(RegressionMetric):

    optimise_direction = 'maximize'


class MinimiseRegressionMetric(RegressionMetric):

    optimise_direction = 'minimise'


def eval_complete(model, train_dataset, val_dataset, test_dataset, criterion, metric: Metric, verbose=False):
    train_results = eval_model(model, train_dataset, criterion, metric)
    if verbose:
        print('\n-- Train Results --')
        train_results.output()
    val_results = eval_model(model, val_dataset, criterion, metric)
    if verbose:
        print('\n-- Val Results --')
        val_results.output()
    test_results= eval_model(model, test_dataset, criterion, metric)
    if verbose:
        print('\n-- Test Results --')
        test_results.output()
    return train_results, val_results, test_results


def eval_model(model, dataset, criterion, metric: Metric):
    model.eval()
    with torch.no_grad():
        all_preds = []
        for bag in dataset.bags:
            bag_pred, _ = model.forward_verbose(bag)
            all_preds.append(bag_pred.detach().cpu())
        labels = list(range(model.n_classes))
        all_preds = torch.cat(all_preds)
        bag_metric = metric.calculate_metric(all_preds, dataset.targets, criterion, labels)
        return bag_metric


def output_results(results: List[Tuple[Metric, Metric, Metric]]):
    results_type = type(results[0][0])
    if results_type == ClassificationMetric:
        output_classification_results(results)
    elif issubclass(results_type, RegressionMetric):
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
    raw_results = np.asarray(raw_results)
    rows = [['Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test Accuracy', 'Test Loss']]
    results_row = []
    for i in range(6):
        values = raw_results[:, i]
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
    raw_results = np.asarray(raw_results)
    rows = [['Train Loss', 'Val Loss', 'Test Loss']]
    results_row = []
    for i in range(3):
        values = raw_results[:, i]
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
