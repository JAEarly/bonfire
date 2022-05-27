import argparse

from bonfire.model.benchmark import get_model_clz
from bonfire.train import DEFAULT_SEEDS
from bonfire.train.benchmark import get_trainer_clz
from bonfire.util import get_device

device = get_device()

DATASET_NAMES = ['crc', 'count_mnist', 'four_mnist', 'musk', 'sival', 'tiger', 'elephant', 'fox']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN', 'MiLstm']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to train.')
    parser.add_argument('-s', '--seeds', default=",".join(str(s) for s in DEFAULT_SEEDS), type=str,
                        help='The seeds for the training. Should be at least as long as the number of repeats.')
    parser.add_argument('-r', '--n_repeats', default=1, type=int, help='The number of models to train (>=1).')
    parser.add_argument('-e', '--n_epochs', default=150, type=int, help='The number of epochs to train for.')
    parser.add_argument('-p', '--patience', default=15, type=int, help='The patience to use during training.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.seeds, args.n_repeats, args.n_epochs, args.patience


def run_training():
    dataset_name, model_name, seeds, n_repeats, n_epochs, patience = parse_args()

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

    train_params = {
        'n_epochs': n_epochs,
        'patience': patience
    }

    print('  Training with params: {:}'.format(train_params))
    print('  Seeds: {:}'.format(seeds))

    if dataset_name in ['tiger', 'elephant', 'fox']:
        trainer = trainer_clz(device, train_params, model_clz, dataset_name)
    else:
        trainer = trainer_clz(device, train_params, model_clz)

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
