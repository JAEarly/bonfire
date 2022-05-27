import torch

from bonfire.model.benchmark import get_model_clz
from bonfire.train import DEFAULT_SEEDS, get_default_save_path
from bonfire.train.benchmark import get_trainer_clz
from bonfire.train.metrics import eval_complete, output_results


def run(dataset_name, model_names):
    print('Getting results for dataset {:s}'.format(dataset_name))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name in model_names:
        print('Running for', model_name)
        run_single(device, dataset_name, model_name)


def run_single(device, dataset_name, model_name):
    results = get_results(device, dataset_name, model_name)
    output_results(results)


def get_results(device, dataset_name, model_name):
    model_clz = get_model_clz(dataset_name, model_name)
    trainer_clz = get_trainer_clz(dataset_name, model_clz)
    trainer = trainer_clz(device, {}, model_clz)

    results = []
    for i in range(len(DEFAULT_SEEDS)):
        seed = DEFAULT_SEEDS[i]
        print('Model {:d}/{:d}; Seed {:d}'.format(i + 1, len(DEFAULT_SEEDS), seed))
        train_dataloader, val_dataloader, test_dataloader = trainer.create_dataloaders(seed=seed)
        model_path, _, _ = get_default_save_path(dataset_name, model_name, repeat=i)
        model = model_clz.load_model(device, model_path)
        repeat_results = eval_complete(model, train_dataloader.dataset, val_dataloader.dataset, test_dataloader.dataset,
                                       trainer.get_criterion(), trainer.metric_clz, verbose=False)
        results.append(repeat_results)
    return results


if __name__ == "__main__":
    # TODO from scripts args
    run("sival", ["InstanceSpaceNN"])
