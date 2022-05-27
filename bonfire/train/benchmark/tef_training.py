from abc import ABC

from bonfire.data.benchmark.tef.tef_dataset import TigerDataset, ElephantDataset, FoxDataset
from bonfire.train.train_base import ClassificationTrainer, NetTrainerMixin, GNNTrainerMixin


def get_trainer_clzs():
    return [TigerNetTrainer, TigerGNNTrainer, ElephantNetTrainer, ElephantGNNTrainer, FoxNetTrainer, FoxGNNTrainer]


class TefTrainer(ClassificationTrainer, ABC):

    def get_default_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'patience': 10,
        }


class TigerTrainer(TefTrainer, ABC):
    dataset_clz = TigerDataset


class ElephantTrainer(TefTrainer, ABC):
    dataset_clz = ElephantDataset


class FoxTrainer(TefTrainer, ABC):
    dataset_clz = FoxDataset


class TigerNetTrainer(NetTrainerMixin, TigerTrainer):
    pass


class TigerGNNTrainer(GNNTrainerMixin, TigerTrainer):
    pass


class ElephantNetTrainer(NetTrainerMixin, ElephantTrainer):
    pass


class ElephantGNNTrainer(GNNTrainerMixin, ElephantTrainer):
    pass


class FoxNetTrainer(NetTrainerMixin, FoxTrainer):
    pass


class FoxGNNTrainer(GNNTrainerMixin, FoxTrainer):
    pass
