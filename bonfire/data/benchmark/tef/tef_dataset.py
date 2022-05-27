import csv

import torch
from sklearn.model_selection import train_test_split

from bonfire.data.mil_dataset import MilDataset
from abc import ABC, abstractmethod


class TefDataset(MilDataset, ABC):

    d_in = 230
    n_classes = 2
    n_expected_dims = 2  # i * f

    def __init__(self, bags, targets):
        super().__init__(bags, targets, None)

    @classmethod
    @property
    @abstractmethod
    def csv_path(cls):
        pass

    @classmethod
    def create_datasets(cls, random_state=12):
        parsed_data = parse_data(cls.csv_path)
        bags, targets = parsed_data

        bags = normalise(bags)

        splits = train_test_split(bags, targets, train_size=0.7, stratify=targets, random_state=random_state)
        train_bags, test_bags, train_targets, test_targets = splits

        splits = train_test_split(test_bags, test_targets, train_size=0.5, stratify=test_targets,
                                  random_state=random_state)
        val_bags, test_bags, val_targets, test_targets = splits

        train_dataset = TefDataset(train_bags, train_targets)
        val_dataset = TefDataset(val_bags, val_targets)
        test_dataset = TefDataset(test_bags, test_targets)

        return train_dataset, val_dataset, test_dataset


class TigerDataset(TefDataset):

    name = "Tiger"
    csv_path = "./data/TEF/tiger.svm"


class ElephantDataset(TefDataset):

    name = "Elephant"
    csv_path = "./data/TEF/elephant.svm"


class FoxDataset(TefDataset):

    name = "Fox"
    csv_path = "./data/TEF/fox.svm"


def parse_data(csv_path):
    bag_data = {}
    bag_targets = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        next(reader)
        for line in reader:
            bag_num, features, instance_target = parse_line(line)
            if bag_num not in bag_data:
                bag_data[bag_num] = []
                bag_targets[bag_num] = 0
            bag_data[bag_num].append(features)
            if instance_target == 1:
                bag_targets[bag_num] = 1

    bags = []
    targets = []
    for bag_name in bag_data.keys():
        bag = bag_data[bag_name]
        bag = torch.stack(bag)
        bags.append(bag)
        target = bag_targets[bag_name]
        targets.append(target)

    return bags, targets


def parse_line(line):
    _, bag_num, label = line[0].split(":")
    bag_num = int(bag_num)
    instance_target = int(label)

    features = []
    for f in line[1:]:
        if ':' in f:
            features.append(float(f.split(":")[1]))
    features = torch.as_tensor(features)
    return bag_num, features, instance_target


def normalise(bags):
    all_instances = torch.cat(bags)
    dataset_mean = torch.mean(all_instances, dim=0)
    dataset_std = torch.std(all_instances, dim=0)
    norm_bags = []
    for bag in bags:
        norm_bag = (bag - dataset_mean) / dataset_std
        norm_bag = torch.as_tensor(norm_bag).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        norm_bags.append(norm_bag)
    return norm_bags


if __name__ == "__main__":
    print("\n -- TIGER -- ")
    for d in TefDataset.create_datasets("tiger"):
        d.summarise()

    print("\n -- ELEPHANT -- ")
    for d in TefDataset.create_datasets("elephant"):
        d.summarise()

    print("\n -- FOX -- ")
    for d in TefDataset.create_datasets("fox"):
        d.summarise()
