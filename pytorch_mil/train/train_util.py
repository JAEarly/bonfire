import random

import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from torch import nn
from torch_geometric.data.data import Data


class GraphDataloader:

    def __init__(self, graph_dataset):
        self.graph_dataset = graph_dataset
        self.n_graphs = len(self.graph_dataset.bags)

    def __iter__(self):
        self.idx = 0
        self.order = list(range(self.n_graphs))
        random.shuffle(self.order)
        return self

    def __next__(self):
        if self.idx >= self.n_graphs:
            raise StopIteration
        o_idx = self.order[self.idx]
        edge_index = self.graph_dataset.edge_indexs[o_idx]
        instances, target = self.graph_dataset[o_idx]
        data = Data(x=instances, edge_index=edge_index)
        self.idx += 1
        return [data], target.unsqueeze(0)

    def __len__(self):
        return self.n_graphs


def eval_complete(model, train_dataloader, val_dataloader, test_dataloader, verbose=False):
    if verbose:
        print('\n-- Train Results --')
    train_results = eval_model(model, train_dataloader, verbose=verbose)
    if verbose:
        print('\n-- Val Results --')
    val_results = eval_model(model, val_dataloader, verbose=verbose)
    if verbose:
        print('\n-- Test Results --')
    test_results = eval_model(model, test_dataloader, verbose=verbose)
    return train_results, val_results, test_results


def eval_model(model, dataloader, verbose=False):
    model.eval()
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss()
        labels = list(range(model.n_classes))
        all_probas = []
        all_targets = []
        for data in dataloader:
            bags, targets = data[0], data[1]
            bag_probas = model(bags)
            all_probas.append(bag_probas.detach().cpu())
            all_targets.append(targets.detach().cpu())
        all_targets = torch.cat(all_targets).long()
        all_probas = torch.cat(all_probas)
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
    return acc, loss


def output_results(results):
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
