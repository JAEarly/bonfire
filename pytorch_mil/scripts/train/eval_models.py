import torch

from pytorch_mil.data import load_datasets
from pytorch_mil.model import load_model
from pytorch_mil.train.metrics import eval_complete, output_results

SEEDS = [868, 207, 702, 999, 119, 401, 74, 9, 741, 744]


def run(dataset_name, model_names):
    print('Getting results for dataset {:s}'.format(dataset_name))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name in model_names:
        print('Running for', model_name)
        run_single(device, dataset_name, model_name)


def run_single(device, dataset_name, model_name):
    results = get_results(device, dataset_name, model_name)
    output_results(results)


def get_results(device, dataset_name, model_name, criterion):
    results = []
    for i in range(len(SEEDS)):
        print('Model {:d}/{:d}'.format(i + 1, len(SEEDS)))
        train_dataloader, val_dataloader, test_dataloader = load_datasets(dataset_name, seed=SEEDS[i])
        model = load_model(device, model_name, dataset_name, repeat=i)
        final_results = eval_complete(model, train_dataloader, val_dataloader, test_dataloader, criterion, verbose=False)
        train_results, val_results, test_results = final_results
        results.append([train_results[0], train_results[1],
                        val_results[0], val_results[1],
                        test_results[0], test_results[1]])
    return results


if __name__ == "__main__":
    # TODO from scripts args
    run("sival", ["InstanceSpaceNN"])
