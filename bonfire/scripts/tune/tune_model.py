import argparse

from bonfire.model.benchmark import get_model_clz
from bonfire.train.benchmark import get_trainer_clz
from bonfire.tune.benchmark import get_tuner_clz
from bonfire.tune.tune_util import setup_study
from bonfire.util import get_device

device = get_device()

DATASET_NAMES = ['crc', 'count_mnist', 'dgr', 'four_mnist', 'masati', 'musk', 'sival', 'tiger', 'elephant', 'fox']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN', 'MiLstm']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL tuning script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to tune.')
    parser.add_argument('-n', '--n_trials', default=100, type=int, help='The number of trials to run when tuning.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.n_trials


def run_tuning(dataset_name, model_name, n_trials):
    print('Starting {:s} tuning'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using dataset {:}'.format(dataset_name))
    print('  Using device {:}'.format(device))

    model_clz = get_model_clz(dataset_name, model_name)
    trainer_clz = get_trainer_clz(dataset_name, model_clz)
    tuner_clz = get_tuner_clz(model_clz)
    print('  Model Class: {:}'.format(model_clz))
    print('  Trainer Class: {:}'.format(trainer_clz))
    print('  Tuner Class: {:}'.format(tuner_clz))

    direction = trainer_clz.metric_clz.optimise_direction
    print('  Direction: {:s}'.format(direction))

    study = setup_study("Optimise-{:s}-{:s}".format(dataset_name, model_name),
                        "{:s}/{:s}".format(dataset_name, model_name),
                        direction=direction)
    tuner = tuner_clz(device)

    print('  Running study with {:d} trials'.format(n_trials))
    study.optimize(tuner, n_trials=n_trials)


if __name__ == "__main__":
    _dataset_name, _model_name, _n_trials = parse_args()
    run_tuning(_dataset_name, _model_name, _n_trials)
