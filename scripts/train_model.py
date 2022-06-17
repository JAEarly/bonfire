import argparse

import wandb

from bonfire.data.benchmark import dataset_names
from bonfire.model.benchmark import model_names
from bonfire.train import DEFAULT_SEEDS
from bonfire.train.trainer import create_trainer_from_names
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_name', choices=model_names, help='The model to train.')
    parser.add_argument('-s', '--seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for the training. Should be at least as long as the number of repeats.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.seeds, args.n_repeats


def run_training():
    dataset_name, model_name, seeds, n_repeats = parse_args()

    # Parse wandb config and get training config for this model
    config = parse_yaml_config("bonfire/bonfire/config/four_mnist_config.yaml")
    training_config = parse_training_config(config['training'], model_name)
    wandb.init(project="MIL-Four_MNIST-Test", config=training_config)

    # Parse seed list
    seeds = [int(s) for s in seeds.split(",")]
    if len(seeds) < n_repeats:
        raise ValueError('Not enough seeds provided for {:d} repeats'.format(n_repeats))
    seeds = seeds[:n_repeats]

    # Create trainer
    trainer = create_trainer_from_names(device, model_name, dataset_name)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))
    print('  Seeds: {:}'.format(seeds))

    # Start training
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
