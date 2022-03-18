import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable


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
        all_preds = []
        all_targets = []
        for data in dataloader:
            bags, targets = data[0], data[1]
            bag_preds = model(bags)
            all_preds.append(bag_preds.detach().cpu())
            all_targets.append(targets.detach().cpu())
        all_targets = torch.cat(all_targets).long()
        all_preds = torch.cat(all_preds)
        if model.n_classes > 1:
            # Classification
            _, all_preds = torch.max(F.softmax(all_preds, dim=1), dim=1)
            acc = accuracy_score(all_targets, all_preds)
            loss = criterion(all_preds, all_targets).item()
            if verbose:
                conf_mat = pd.DataFrame(
                    confusion_matrix(all_targets, all_preds, labels=labels),
                    index=pd.Index(labels, name='Actual'),
                    columns=pd.Index(labels, name='Predicted')
                )
                print(' Acc: {:.3f}'.format(acc))
                print('Loss: {:.3f}'.format(loss))
                print(conf_mat)
        else:
            # TODO sort this out, it's messy
            # Regression
            acc = criterion(all_preds, all_targets).item()
            loss = criterion(all_preds, all_targets).item()
            if verbose:
                print(' Acc: {:.3f}'.format(acc))
                print('Loss: {:.3f}'.format(loss))
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