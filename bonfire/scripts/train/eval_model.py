import argparse

import numpy as np

from bonfire.model.benchmark import get_model_clz
from bonfire.scripts.train.train_model import DATASET_NAMES, MODEL_NAMES
from bonfire.train import DEFAULT_SEEDS, get_default_save_path
from bonfire.train.benchmark import get_trainer_clz
from bonfire.train.metrics import eval_complete, output_results
from bonfire.util import get_device

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_names', choices=MODEL_NAMES, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to evaluate (>=1).')
    parser.add_argument('-s', '--seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for dataset generation. Should be at least as long as the number of repeats.'
                             'Should match the seeds used during training.')
    parser.add_argument('-v', '--verbose', default=False, action="store_true")
    args = parser.parse_args()
    return args.dataset_name, args.model_names, args.n_repeats, args.seeds, args.verbose


def run_evaluation():
    dataset_name, model_names, n_repeats, seeds, verbose = parse_args()
    # Parse seed list
    seeds = [int(s) for s in seeds.split(",")]
    if len(seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    seeds = seeds[:n_repeats]

    print('Getting results for dataset {:s}'.format(dataset_name))
    results = np.empty((len(model_names), n_repeats, 3), dtype=object)
    print('Running for models: {:}'.format(model_names))
    for model_idx, model_name in enumerate(model_names):
        print('  Evaluating {:s}'.format(model_name))
        model_clz = get_model_clz(dataset_name, model_name)
        trainer_clz = get_trainer_clz(dataset_name, model_clz)
        trainer = trainer_clz(device, model_clz, dataset_name)
        if n_repeats == 1:
            model_results = evaluate_single(dataset_name, model_clz, trainer, seeds[0], verbose)
        else:
            model_results = evaluate_multiple(dataset_name, model_clz, trainer, seeds, verbose)
        results[model_idx, :, :] = model_results
    output_results(model_names, results)


def evaluate_single(dataset_name, model_clz, trainer, seed, verbose):
    print('    Evaluating default model, seed: {:}'.format(seed))
    model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__)
    results = get_results(trainer, seed, model_clz, model_path, verbose)
    return results


def evaluate_multiple(dataset_name, model_clz, trainer, seeds, verbose):
    results = np.empty((len(seeds), 3), dtype=object)
    for i in range(len(seeds)):
        seed = seeds[i]
        print('    Model {:d}/{:d}; Seed {:d}'.format(i + 1, len(DEFAULT_SEEDS), seed))
        model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=i)
        results[i, :] = get_results(trainer, seed, model_clz, model_path, verbose)
    return results


def get_results(trainer, seed, model_clz, model_path, verbose):
    results = np.empty((1, 3), dtype=object)
    train_dataset, val_dataset, test_dataset = trainer.load_datasets(seed)
    model = model_clz.load_model(device, model_path)
    results_list = eval_complete(model, train_dataset, val_dataset, test_dataset, trainer.metric_clz, verbose=verbose)
    results[0, :] = results_list
    return results


if __name__ == "__main__":
    run_evaluation()
