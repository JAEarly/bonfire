from abc import ABC, abstractmethod
from types import SimpleNamespace

from pytorch_mil.model import models


class Tuner(ABC):

    def __init__(self, device, model_clz, trainer_clz, dataset_name):
        self.device = device
        self.model_clz = model_clz
        self.trainer_clz = trainer_clz
        self.dataset_name = dataset_name

    @abstractmethod
    def generate_model_yobj(self, trial):
        pass

    def create_trainer(self, model_yobj):
        if self.dataset_name in ['tiger', 'elephant', 'fox']:
            return self.trainer_clz(self.device, self.model_clz, self.dataset_name, model_yobj=model_yobj)
        return self.trainer_clz(self.device, self.model_clz, model_yobj=model_yobj)

    @staticmethod
    def suggest_layers(trial, layer_name, min_n_layers, max_n_layers, layer_options):
        n_layers = trial.suggest_int('n_{:s}'.format(layer_name), min_n_layers, max_n_layers)
        layers_values = []
        for i in range(n_layers):
            layers_values.append(trial.suggest_categorical('{:s}_{:d}'.format(layer_name, i), layer_options))
        return layers_values

    def generate_encoder(self, trial, dropout):
        d_enc = trial.suggest_categorical('d_enc', [64, 128, 256, 512])
        encoder = SimpleNamespace(
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

    def __call__(self, trial):
        model_yobj = self.generate_model_yobj(trial)
        trainer = self.create_trainer(model_yobj)
        model, _, _, test_results, early_stopped = trainer.train_single(save_model=False, show_plot=False,
                                                                        verbose=False, trial=trial)
        test_acc = test_results[0]
        trial.set_user_attr('early_stopped', early_stopped)
        return test_acc


class EmbeddingSpaceNNTuner(Tuner):

    def __init__(self, device, trainer_clz, dataset_name):
        super().__init__(device, base_models.EmbeddingSpaceNN, trainer_clz, dataset_name)

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


class InstanceSpaceNNTuner(Tuner):

    def __init__(self, device, trainer_clz, dataset_name):
        super().__init__(device, base_models.InstanceSpaceNN, trainer_clz, dataset_name)

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


class AttentionNNTuner(Tuner):

    def __init__(self, device, trainer_clz, dataset_name):
        super().__init__(device, base_models.AttentionNN, trainer_clz, dataset_name)

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


class MultiHeadAttentionNNTuner(Tuner):

    def __init__(self, device, trainer_clz, dataset_name):
        super().__init__(device, base_models.MultiHeadAttentionNN, trainer_clz, dataset_name)

    def generate_model_yobj(self, trial):
        train_params = self.generate_train_params(trial)
        encoder = self.generate_encoder(trial, train_params.dropout)
        aggregator = SimpleNamespace(
            d_in=encoder.d_out,
            ds_hid=self.suggest_layers(trial, "agg_hid", 0, 2, [64, 128, 256]),
            dropout=train_params.dropout,
            d_attn=trial.suggest_categorical('d_attn', [64, 128, 256]),
            n_heads=trial.suggest_categorical('n_head', [2, 3, 4])
        )
        train_params = SimpleNamespace(
            lr=trial.suggest_categorical('lr', [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
            wd=trial.suggest_categorical('wd', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]),
        )
        model_yobj = SimpleNamespace(
            Encoder=encoder,
            Aggregator=aggregator,
            TrainParams=train_params,
        )
        return model_yobj


class ClusterGNNTuner(Tuner):

    def __init__(self, device, trainer_clz, dataset_name):
        super().__init__(device, base_models.ClusterGNN, trainer_clz, dataset_name)

    def generate_model_yobj(self, trial):
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
