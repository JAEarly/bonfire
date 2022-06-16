from .mnist.mnist_bags import FourMnistBagsDataset


def get_dataset_clz(dataset_name):
    if dataset_name == 'crc':
        pass
    elif dataset_name == 'count_mnist':
        pass
    elif dataset_name == 'dgr':
        pass
    elif dataset_name == 'four_mnist':
        return FourMnistBagsDataset
    elif dataset_name == 'masati':
        pass
    elif dataset_name == 'musk':
        pass
    elif dataset_name == 'sival':
        pass
    elif dataset_name in ['tiger', 'elephant', 'fox']:
        pass
    else:
        raise ValueError('No dataset registered with name {:s}'.format(dataset_name))
