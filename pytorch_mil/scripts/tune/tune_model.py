import argparse

from pytorch_mil.tune import sival_tuning, mnist_tuning  #crc_tuning, mnist_tuning, sival_tuning
from pytorch_mil.tune.tune_util import setup_study
from pytorch_mil.util.misc_util import get_device

device = get_device()

DATASET_NAMES = ['crc', 'mnist', 'musk', 'sival', 'tef']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN']


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL tuning script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to tune.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name


def get_tuner_clz(dataset_name, model_name):
    if dataset_name == 'crc':
        return crc_tuning.get_tuner_from_name(model_name)
    if dataset_name == 'mnist':
        return mnist_tuning.get_tuner_from_name(model_name)
    if dataset_name == 'musk':
        raise NotImplementedError
    if dataset_name == 'sival':
        return sival_tuning.get_tuner_from_name(model_name)
    if dataset_name == 'tef':
        raise NotImplementedError
    raise ValueError("No tuner found for datasets {:s}".format(dataset_name))


def run_tuning(dataset_name, model_name):
    print('Starting {:s} tuning'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    tuner_clz = get_tuner_clz(dataset_name, model_name)
    study = setup_study("Optimise-{:s}".format(model_name))
    tuner = tuner_clz(device)
    study.optimize(tuner, n_trials=100)


if __name__ == "__main__":
    _dataset_name, _model_name = parse_args()
    run_tuning(_dataset_name, _model_name)
