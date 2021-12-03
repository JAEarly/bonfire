import argparse

from pytorch_mil.train import get_trainer_clz
from pytorch_mil.tune.tune_util import setup_study, get_tuner_clz
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


def make_tuner(dataset_name, model_name):
    trainer_clz = get_trainer_clz(dataset_name, model_name)
    tuner_clz = get_tuner_clz(model_name)
    print(trainer_clz, tuner_clz)
    return tuner_clz(device, trainer_clz)


def run_tuning(dataset_name, model_name):
    print('Starting {:s} tuning'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    study = setup_study("Optimise-{:s}".format(model_name))
    tuner = make_tuner(dataset_name, model_name)
    study.optimize(tuner, n_trials=100)


if __name__ == "__main__":
    _dataset_name, _model_name = parse_args()
    run_tuning(_dataset_name, _model_name)
