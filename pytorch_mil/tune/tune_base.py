from abc import ABC, abstractmethod


class Tuner(ABC):

    def __init__(self, device, model_clz):
        self.device = device
        self.model_clz = model_clz

    @abstractmethod
    def generate_model_yobj(self, trial):
        pass

    @abstractmethod
    def create_trainer(self, model_yobj):
        pass

    @staticmethod
    def suggest_layers(trial, layer_name, min_n_layers, max_n_layers, layer_options):
        n_layers = trial.suggest_int('n_{:s}'.format(layer_name), min_n_layers, max_n_layers)
        layers_values = []
        for i in range(n_layers):
            layers_values.append(trial.suggest_categorical('{:s}_{:d}'.format(layer_name, i), layer_options))
        return layers_values

    def __call__(self, trial):
        model_yobj = self.generate_model_yobj(trial)
        trainer = self.create_trainer(model_yobj)
        model, _, _, test_results, early_stopped = trainer.train_single(save_model=False, show_plot=False,
                                                                        verbose=False, trial=trial)
        test_acc = test_results[0]
        trial.set_user_attr('early_stopped', early_stopped)
        return test_acc
