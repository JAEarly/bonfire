from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod


# TODO Add abstract method to compute the mean.
class MilDataset(Dataset, ABC):

    def __init__(self, bags, targets, instance_targets):
        super(Dataset, self).__init__()
        self.bags = bags
        self.targets = torch.as_tensor(targets).float()
        self.instance_targets = instance_targets

    @classmethod
    @abstractmethod
    def create_datasets(cls, random_state=12):
        pass

    def summarise(self, out_clz_dist=True):
        clz_dist = Counter(np.asarray(self.targets))
        print('- MIL Dataset Summary -')
        print(' {:d} bags'.format(len(self.bags)))

        if out_clz_dist:
            print(' Class Distribution')
            for clz in sorted(clz_dist.keys()):
                print('  {:d}: {:d} ({:.2f}%)'.format(int(clz), clz_dist[clz], clz_dist[clz]/len(self)*100))

        bag_sizes = [len(b) for b in self.bags]
        print(' Bag Sizes')
        print('  Min: {:d}'.format(min(bag_sizes)))
        print('  Avg: {:.1f}'.format(np.mean(bag_sizes)))
        print('  Max: {:d}'.format(max(bag_sizes)))

    def get_bag_verbose(self, index):
        bag = self.bags[index]
        target = self.targets[index]
        instance_targets = self.instance_targets[index] if self.instance_targets is not None else None
        return bag, target, instance_targets

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        target = self.targets[index]
        return bag, target
