import argparse

import wandb

from bonfire.model.benchmark import get_model_clz
from bonfire.train import DEFAULT_SEEDS
from bonfire.train.benchmark import get_trainer_clz
from bonfire.util import get_device
import yaml


device = get_device()

DATASET_NAMES = ['crc', 'count_mnist', 'dgr', 'four_mnist', 'masati', 'musk', 'sival', 'tiger', 'elephant', 'fox']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN', 'MiLstm']


def parse_config(path, model_name):
    stream = open(path, 'r')
    config = {}
    model_override_config = {}
    for config_name, params in yaml.safe_load(stream).items():
        if config_name == 'default':
            config = params
        elif config_name == model_name:
            model_override_config = params
    print('Default:', config)

    for param_name, param_value in model_override_config.items():
        config[param_name] = param_value
        print('Override', param_name)

    print('New:', config)

    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to train.')
    parser.add_argument('-s', '--seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for the training. Should be at least as long as the number of repeats.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    parser.add_argument('-e', '--n_epochs', type=int, help='The number of epochs to train for.')
    parser.add_argument('-p', '--patience', type=int, help='The patience to use during training.')
    parser.add_argument('-i', '--patience_interval', type=int, help='The patience interval to use during training.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.seeds, args.n_repeats, \
        args.n_epochs, args.patience, args.patience_interval


def run_training():
    dataset_name, model_name, seeds, n_repeats, n_epochs, patience, patience_interval = parse_args()

    config = parse_config("bonfire/bonfire/config/four_mnist_config.yaml", model_name)

    wandb.init(project="MIL-Four_MNIST-Test", config=config)

    # Parse seed list
    seeds = [int(s) for s in seeds.split(",")]
    if len(seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    seeds = seeds[:n_repeats]

    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    model_clz = get_model_clz(dataset_name, model_name)
    trainer_clz = get_trainer_clz(dataset_name, model_clz)
    print('  Model Class: {:}'.format(model_clz))
    print('  Trainer Class: {:}'.format(trainer_clz))

    train_params_override = {}
    if n_epochs is not None:
        train_params_override['n_epochs'] = n_epochs
    if patience is not None:
        train_params_override['patience'] = patience
    if patience_interval is not None:
        train_params_override['patience_interval'] = patience_interval

    print('  Training params override: {:}'.format(train_params_override))
    print('  Seeds: {:}'.format(seeds))

    trainer = trainer_clz(device, model_clz, dataset_name, train_params_override=train_params_override)

    if n_repeats > 1:
        print('  Training using multiple trainer')
        trainer.train_multiple(n_repeats=n_repeats, seeds=seeds)
    elif n_repeats == 1:
        print('  Training using single trainer')
        trainer.train_single(seed=seeds[0])
    else:
        raise ValueError("Invalid number of repeats for training: {:d}".format(n_repeats))


if __name__ == "__main__":
    run_training()
