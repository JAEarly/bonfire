from .sival import sival_dataset
from torch.utils.data import DataLoader


def load_datasets(dataset_name, seed=None):
    if dataset_name == 'sival':
        train_dataset, val_dataset, test_dataset = sival_dataset.create_datasets(random_state=seed)
    # if dataset_name == 'musk':
    #     train_dataset, val_dataset, test_dataset = musk_dataset.create_datasets(musk_two=False, random_state=seed)
    # elif dataset_name in ['tiger', 'elephant', 'fox']:
    #     train_dataset, val_dataset, test_dataset = tef_dataset.create_datasets(dataset_name, random_state=seed)
    else:
        raise ValueError('Invalid dataset name: {:s}'.format(dataset_name))
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1, num_workers=1)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=1)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)
    return train_dataloader, val_dataloader, test_dataloader
