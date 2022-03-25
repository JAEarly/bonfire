import argparse

from pytorch_mil.model.benchmark import get_model_clz
from pytorch_mil.train.benchmark import get_trainer_clz
from pytorch_mil.util import get_device

device = get_device()

DATASET_NAMES = ['crc', 'count_mnist', 'four_mnist', 'musk', 'sival', 'tiger', 'elephant', 'fox']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN', 'MiLstm']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to train.')
    parser.add_argument('-s', '--seed', default=5, help='The seed for the training.')
    parser.add_argument('-r', '--n_repeats', default=1, help='The number of models to train (>=1).')
    parser.add_argument('-e', '--n_epochs', default=150, help='The number of epochs to train for.')
    parser.add_argument('-p', '--patience', default=15, help='The patience to use during training.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.seed, args.n_repeats, args.n_epochs, args.patience


def run_training():
    dataset_name, model_name, seed, n_repeats, n_epochs, patience = parse_args()

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
    print('  Seed: {:d}'.format(seed))

    if dataset_name in ['tiger', 'elephant', 'fox']:
        trainer = trainer_clz(device, train_params, model_clz, dataset_name)
    else:
        trainer = trainer_clz(device, train_params, model_clz)

    if n_repeats > 1:
        print('  Training using multiple trainer')
        trainer.train_multiple(n_repeats=n_repeats, seed=seed)
    elif n_repeats == 1:
        print('  Training using single trainer')
        trainer.train_single(seed=seed)
    else:
        raise ValueError("Invalid number of repeats for training: {:d}".format(n_repeats))


if __name__ == "__main__":
    run_training()
