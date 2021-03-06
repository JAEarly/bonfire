import inspect
from abc import ABC, abstractmethod


class Tuner(ABC):

    model_clz = NotImplemented

    def __init__(self, device, dataset_name):
        self.device = device
        self.dataset_name = dataset_name

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # All tuner concrete classes need to have a class attribute model_clz define
        if cls.model_clz is NotImplemented and not inspect.isabstract(cls):
            raise NotImplementedError('No model_clz defined for tuner {:}.'.format(cls))

    @abstractmethod
    def generate_train_params(self, trial):
        pass

    @abstractmethod
    def generate_model_params(self, trial):
        pass

    @abstractmethod
    def create_trainer(self, train_params, model_params):
        pass

    @staticmethod
    def suggest_layers(trial, layer_name, param_name, min_n_layers, max_n_layers, layer_options):
        n_layers = trial.suggest_int('n_{:s}'.format(layer_name), min_n_layers, max_n_layers)
        layers_values = []
        for i in range(n_layers):
            layers_values.append(trial.suggest_categorical('{:s}_{:d}'.format(param_name, i), layer_options))
        return layers_values

    def __call__(self, trial):
        train_params = self.generate_train_params(trial)
        model_params = self.generate_model_params(trial)
        trainer = self.create_trainer(train_params, model_params)
        model, _, _, test_results, early_stopped = trainer.train_single(save_model=False, show_plot=False,
                                                                        verbose=False, trial=trial)
        test_metric = test_results.key_metric()
        trial.set_user_attr('early_stopped', early_stopped)
        return test_metric
