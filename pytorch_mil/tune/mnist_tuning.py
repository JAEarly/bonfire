from abc import ABC
from types import SimpleNamespace

from overrides import overrides

from pytorch_mil.model import base_models
from pytorch_mil.train.mnist_training import MnistNetTrainer, MnistGNNTrainer
from pytorch_mil.tune.tune_base import Tuner


def get_tuner_from_name(model_name):
    if model_name == 'InstanceSpaceNN':
        return MnistInstanceSpaceNNTuner
    if model_name == 'EmbeddingSpaceNN':
        return MnistEmbeddingSpaceNNTuner
    if model_name == 'AttentionNN':
        return MnistAttentionNNTuner
    if model_name == 'ClusterGNN':
        return MnistGNNTuner
    raise ValueError("No MNIST tuner implemented for model name {:s}".format(model_name))


class MnistTuner(Tuner, ABC):

    def create_trainer(self, model_yobj):
        return MnistNetTrainer(self.device, self.model_clz, model_yobj=model_yobj)

    def generate_encoder(self, trial, dropout):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        encoder = SimpleNamespace(
            type='conv',
            d_out=d_enc,
            ds_hid=self.suggest_layers(trial, "enc_hid", 0, 2, [64, 128, 256]),
            dropout=dropout
        )
        return encoder

    @staticmethod
    def generate_train_params(trial):
        train_params = SimpleNamespace(
            lr=trial.suggest_categorical('lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            wd=trial.suggest_categorical('wd', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
            dropout=trial.suggest_float('dropout', 0, 0.5, step=0.05),
        )
        return train_params


class MnistInstanceSpaceNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, base_models.InstanceSpaceNN)

    def generate_model_yobj(self, trial):
        train_params = self.generate_train_params(trial)
        encoder = self.generate_encoder(trial, train_params.dropout)
        aggregator = SimpleNamespace(
            d_in=encoder.d_out,
            ds_hid=self.suggest_layers(trial, "agg_hid", 0, 2, [64, 128, 256]),
            dropout=train_params.dropout,
            agg_func_name=trial.suggest_categorical('agg_func', ['mean', 'max'])
        )
        model_yobj = SimpleNamespace(
            Encoder=encoder,
            Aggregator=aggregator,
            TrainParams=train_params,
        )
        return model_yobj


class MnistEmbeddingSpaceNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, base_models.EmbeddingSpaceNN)

    def generate_model_yobj(self, trial):
        train_params = self.generate_train_params(trial)
        encoder = self.generate_encoder(trial, train_params.dropout)
        aggregator = SimpleNamespace(
            d_in=encoder.d_out,
            ds_hid=self.suggest_layers(trial, "agg_hid", 0, 2, [64, 128, 256]),
            dropout=train_params.dropout,
            agg_func_name=trial.suggest_categorical('agg_func', ['mean', 'max'])
        )
        model_yobj = SimpleNamespace(
            Encoder=encoder,
            Aggregator=aggregator,
            TrainParams=train_params,
        )
        return model_yobj


class MnistAttentionNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, base_models.AttentionNN)

    def generate_model_yobj(self, trial):
        train_params = self.generate_train_params(trial)
        encoder = self.generate_encoder(trial, train_params.dropout)
        aggregator = SimpleNamespace(
            d_in=encoder.d_out,
            ds_hid=self.suggest_layers(trial, "agg_hid", 0, 2, [64, 128, 256]),
            dropout=train_params.dropout,
            d_attn=trial.suggest_categorical('d_attn', [64, 128, 256])
        )
        model_yobj = SimpleNamespace(
            Encoder=encoder,
            Aggregator=aggregator,
            TrainParams=train_params,
        )
        return model_yobj


class MnistGNNTuner(MnistTuner):

    def __init__(self, device):
        super().__init__(device, base_models.ClusterGNN)

    @overrides
    def create_trainer(self, model_yobj):
        return MnistGNNTrainer(self.device, self.model_clz, model_yobj=model_yobj)

    def generate_model_params(self, trial):
        train_params = self.generate_train_params(trial)
        encoder = self.generate_encoder(trial, train_params.dropout)
        gnn = SimpleNamespace(
            d_enc=encoder.d_out,
            d_gnn=trial.suggest_categorical('d_gnn', [64, 128, 256, 512]),
            ds_gnn_hid=self.suggest_layers(trial, "gnn_hid", 0, 2, [64, 128, 256]),
            ds_fc_hid=self.suggest_layers(trial, "fc_hid", 0, 2, [64, 128, 256]),
        )
        model_yobj = SimpleNamespace(
            Encoder=encoder,
            GNN=gnn,
            TrainParams=train_params,
        )
        return model_yobj
