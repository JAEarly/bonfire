from abc import ABC, abstractmethod

import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from typing import List, Tuple


class Metrics(ABC):

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


class ClassificationMetrics(Metrics):

    optimise_direction = 'maximise'

    def __init__(self, accuracy, loss):
        self.accuracy = accuracy
        self.loss = loss

    def key_metric(self):
        return self.accuracy


class RegressionMetrics(Metrics):

    optimise_direction = 'minimise'

    def __init__(self, loss):
        self.loss = loss

    def key_metric(self):
        return self.loss


def eval_complete(model, train_dataloader, val_dataloader, test_dataloader, criterion, verbose=False):
    if verbose:
        print('\n-- Train Results --')
    train_results = eval_model(model, train_dataloader, criterion, verbose=verbose)
    if verbose:
        print('\n-- Val Results --')
    val_results = eval_model(model, val_dataloader, criterion, verbose=verbose)
    if verbose:
        print('\n-- Test Results --')
    test_results = eval_model(model, test_dataloader, criterion, verbose=verbose)
    return train_results, val_results, test_results


def eval_model(model, dataloader, criterion, verbose=False):
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
        if model.n_classes > 1:
            # Classification
            _, all_preds = torch.max(F.softmax(all_probas, dim=1), dim=1)
            acc = accuracy_score(all_targets, all_preds)
            loss = criterion(all_probas, all_targets).item()
            if verbose:
                conf_mat = pd.DataFrame(
                    confusion_matrix(all_targets, all_preds, labels=labels),
                    index=pd.Index(labels, name='Actual'),
                    columns=pd.Index(labels, name='Predicted')
                )
                print(' Acc: {:.3f}'.format(acc))
                print('Loss: {:.3f}'.format(loss))
                print(conf_mat)
            return ClassificationMetrics(acc, loss)
        else:
            # Regression
            loss = criterion(all_probas, all_targets).item()
            if verbose:
                print('Loss: {:.3f}'.format(loss))
            return RegressionMetrics(loss)


def output_results(results: List[Tuple[Metrics, Metrics, Metrics]]):
    results_type = type(results[0][0])
    if results_type == ClassificationMetrics:
        output_classification_results(results)
    elif results_type == RegressionMetrics:
        output_regression_results(results)
    else:
        raise NotImplementedError('No results output for metrics {:}'.format(results_type))


def output_classification_results(results: List[Tuple[ClassificationMetrics, ClassificationMetrics,
                                                      ClassificationMetrics]]):
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


def output_regression_results(results: List[Tuple[RegressionMetrics, RegressionMetrics, RegressionMetrics]]):
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
