import argparse

import numpy as np
import torch
import wandb

from bonfire.data.benchmark import dataset_names
from bonfire.model.benchmark import model_names
from bonfire.train import get_default_save_path
from bonfire.train.metrics import eval_complete, output_results
from bonfire.train.trainer import create_trainer_from_names
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_names', choices=model_names, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=5, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_names, args.n_repeats


def run_evaluation():
    dataset_name, models, n_repeats = parse_args()

    print('Getting results for dataset {:s}'.format(dataset_name))
    print('Running for models: {:}'.format(models))

    results = np.empty((len(models), n_repeats, 3), dtype=object)
    for model_idx, model_name in enumerate(models):
        print('  Evaluating {:s}'.format(model_name))

        # Download models
        # TODO this should be a util function
        api = wandb.Api()
        for repeat_idx in range(n_repeats):
            artifact = api.artifact('Train_{:s}/{:s}_{:d}.pkl:latest'.format(dataset_name, model_name, repeat_idx))
            artifact.file("models/{:s}/{:s}".format(dataset_name, model_name))

        config = parse_yaml_config(dataset_name)
        training_config = parse_training_config(config['training'], model_name)
        wandb.init(
            config=training_config,
            reinit=True,
        )

        trainer = create_trainer_from_names(device, model_name, dataset_name)
        model_results = evaluate(n_repeats, trainer)
        results[model_idx, :, :] = model_results
    output_results(models, results, sort=False)


def evaluate(n_repeats, trainer, random_state=5):
    results_arr = np.empty((n_repeats, 3), dtype=object)
    r = 0
    for train_dataset, val_dataset, test_dataset in trainer.dataset_clz.create_datasets(random_state=random_state):
        print('Repeat {:d}/{:d}'.format(r + 1, n_repeats))

        train_dataloader = trainer.create_dataloader(train_dataset, True, 0)
        val_dataloader = trainer.create_dataloader(val_dataset, False, 0)
        test_dataloader = trainer.create_dataloader(test_dataset, False, 0)
        model = load_model(trainer.dataset_clz, trainer.model_clz, modifier=r)
        results_list = eval_complete(model, train_dataloader, val_dataloader, test_dataloader,
                                     trainer.metric_clz, verbose=False)
        results_arr[r, :] = results_list
        r += 1
        if r == n_repeats:
            break
    return results_arr


# TODO should be util
def load_model(dataset_clz, model_clz, modifier=None):
    path, _, _ = get_default_save_path(dataset_clz.name, model_clz.name, modifier=modifier)
    model = model_clz(device)
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    run_evaluation()
