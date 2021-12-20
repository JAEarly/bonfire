import argparse

from pytorch_mil.model import models
from pytorch_mil.train import get_trainer_clz
from pytorch_mil.util.misc_util import get_device

device = get_device()

DATASET_NAMES = ['crc', 'mnist', 'musk', 'sival', 'tiger', 'elephant', 'fox']
MODEL_NAMES = ['InstanceSpaceNN', 'EmbeddingSpaceNN', 'AttentionNN', 'MultiHeadAttentionNN', 'ClusterGNN']


def parse_args():
    # TODO add arguments for seed(s) and n_repeats
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=DATASET_NAMES, help='The dataset to use.')
    parser.add_argument('model_name', choices=MODEL_NAMES, help='The model to train.')
    parser.add_argument('-m', '--multiple', action='store_true', default=False,
                        help='Train multiple models rather than just one.')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.multiple


def run_training(dataset_name, model_name, multiple=False):
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:s}'.format("multiple models" if multiple else "single model"))
    model_clz = models.get_model_clz_from_name(model_name)
    trainer_clz = get_trainer_clz(dataset_name, model_name)

    if dataset_name in ['tiger', 'elephant', 'fox']:
        trainer = trainer_clz(device, model_clz, dataset_name)
    else:
        trainer = trainer_clz(device, model_clz)

    if multiple:
        trainer.train_multiple()
    else:
        trainer.train_single()


if __name__ == "__main__":
    _dataset_name, _model_name, _multiple = parse_args()
    run_training(_dataset_name, _model_name, multiple=_multiple)