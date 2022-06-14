from abc import ABC

from bonfire.model.benchmark import dgr_models
from bonfire.train.benchmark.dgr_training import DgrNetTrainer
from bonfire.tune.tune_base import Tuner


def get_tuner_clzs():
    return [DgrInstanceSpaceNNTuner, DgrEmbeddingSpaceNNTuner]


class DgrTuner(Tuner, ABC):

    def __init__(self, device):
        super().__init__(device, "dgr")

    def create_trainer(self, train_params_override, model_params):
        return DgrNetTrainer(self.device, self.model_clz, self.dataset_name, model_params=model_params,
                             train_params_override=train_params_override)

    def generate_train_params(self, trial):
        lr = trial.suggest_categorical('lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
        wd = trial.suggest_categorical('wd', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
        train_params = {
            'lr': lr,
            'weight_decay': wd,
        }
        return train_params


class DgrInstanceSpaceNNTuner(DgrTuner):

    model_clz = dgr_models.DgrInstanceSpaceNN

    def generate_model_params(self, trial):
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        agg_func_name = trial.suggest_categorical('agg_func', ['mean', 'max', 'sum'])
        model_params = {
            'dropout': dropout,
            'agg_func_name': agg_func_name,
        }
        return model_params


class DgrEmbeddingSpaceNNTuner(DgrTuner):

    model_clz = dgr_models.DgrEmbeddingSpaceNN

    def generate_model_params(self, trial):
        dropout = trial.suggest_float('dropout', 0, 0.5, step=0.05)
        agg_func_name = trial.suggest_categorical('agg_func', ['mean', 'max'])
        model_params = {
            'dropout': dropout,
            'agg_func_name': agg_func_name,
        }
        return model_params
