import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_mil.data.mil_dataset import MilDataset


def load_mnist(train):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    return MNIST('data/', train=train, download=True, transform=transform)


def split_mnist_datasets(random_state=12):
    # Load original MNIST data
    train_mnist_dataset = load_mnist(train=True)
    test_mnist_dataset = load_mnist(train=False)

    # Split original test into validation and test to ensure no overlap when creating bags
    split_size = int(len(test_mnist_dataset)/2)
    val_mnist_dataset, test_mnist_dataset = random_split(test_mnist_dataset, [split_size]*2,
                                                         generator=torch.Generator().manual_seed(random_state))

    val_mnist_dataset.data = val_mnist_dataset.dataset.data[val_mnist_dataset.indices]
    val_mnist_dataset.targets = val_mnist_dataset.dataset.targets[val_mnist_dataset.indices]
    test_mnist_dataset.data = test_mnist_dataset.dataset.data[test_mnist_dataset.indices]
    test_mnist_dataset.targets = test_mnist_dataset.dataset.targets[test_mnist_dataset.indices]
    return train_mnist_dataset, val_mnist_dataset, test_mnist_dataset


def get_class_idxs(original_dataset, target_clzs):
    selected_idxs = []
    for idx, target in enumerate(original_dataset.targets):
        if target in target_clzs:
            selected_idxs.append(idx)
    return selected_idxs


def show_bag(bag):
    fig, axes = plt.subplots(nrows=4, ncols=4)
    for i, instance in enumerate(bag):
        axes[i//4][i % 4].imshow(instance.permute(1, 2, 0), cmap='gray')
    plt.show()


def show_instance(instance):
    fig, axis = plt.subplots(nrows=1, ncols=1)
    axis.imshow(instance, cmap='gray')
    plt.show()


# TODO these should all inherit from an abstract MNISTMilDataset base class
class SingleDigitMnistBagsDataset(MilDataset):

    name = 'SingleDigitMNIST'
    d_in = 800
    n_expected_dims = 4  # i x c x h x w
    n_classes = 2

    @classmethod
    def create_datasets(cls, mean_bag_size=30, var_bag_size=2, num_train_bags=2500, num_test_bags=1000,
                        random_state=None):
        if random_state is not None:
            np.random.seed(seed=random_state)
        train_mnist_dataset, val_mnist_dataset, test_mnist_dataset = split_mnist_datasets(random_state=random_state)
        train_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_train_bags, train_mnist_dataset)
        val_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, val_mnist_dataset)
        test_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, test_mnist_dataset)
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def _create_dataset(cls, mean_bag_size, var_bag_size, num_bags, original_dataset, discrim_prob=0.1):
        # Split original data into relevant distributions
        # Clz 0 - Non Discrim: N/A    Discrim: 0 to 8
        # Clz 1 - Non Discrim: 0 to 8 Discrim: 9

        zero_to_eight_idxs = get_class_idxs(original_dataset, list(range(9)))
        nine_idxs = get_class_idxs(original_dataset, [9])

        zero_to_eight_data = original_dataset.data[zero_to_eight_idxs]
        nine_data = original_dataset.data[nine_idxs]

        zero_to_eight_targets = [0] * len(zero_to_eight_idxs)
        nine_targets = [1] * len(nine_idxs)

        zero_to_eight_dist = list(zip(zero_to_eight_data, zero_to_eight_targets))
        nine_dist = list(zip(nine_data, nine_targets))

        clz_0_dists = (None, zero_to_eight_dist)
        clz_1_dists = (zero_to_eight_dist, nine_dist)
        clz_dists = [clz_0_dists, clz_1_dists]

        clz_weights = [0.5] * 2
        clz_discrim_probas = [1.0, discrim_prob]
        bags, targets, instance_targets = cls._create_bags(clz_dists, clz_weights, clz_discrim_probas,
                                                           mean_bag_size, var_bag_size, num_bags)
        return cls(bags, targets, instance_targets)

    @classmethod
    def _create_bags(cls, clz_dists, clz_weights, clz_discrim_probas, mean_bag_size, var_bag_size, num_bags):
        bags = []
        targets = []
        instance_targets = []
        for _ in range(num_bags):
            bag_size = int(np.round(np.random.normal(loc=mean_bag_size, scale=var_bag_size, size=1)))
            if bag_size < 2:
                bag_size = 2
            selected_clz = np.random.choice(range(len(clz_weights)), p=clz_weights)
            non_discrim_dist, discrim_dist = clz_dists[selected_clz]
            discrim_proba = clz_discrim_probas[selected_clz]
            bag, target, bag_instance_targets = cls._create_bag(non_discrim_dist, discrim_dist, discrim_proba, bag_size,
                                                                selected_clz)
            bags.append(bag)
            targets.append(target)
            instance_targets.append(bag_instance_targets)
        return bags, targets, instance_targets

    @classmethod
    def _create_bag(cls, non_discrim_dist, discrim_dist, discrim_proba, bag_size, target):
        bag = []
        instance_targets = []
        bag_target = None

        while len(bag) < bag_size or bag_target is None or bag_target != target:
            if len(bag) == bag_size:
                del bag[0]
                del instance_targets[0]

            # Select if this instance is going to be discriminatory or non-discriminatory
            dist = discrim_dist if np.random.random(1)[0] < discrim_proba else non_discrim_dist

            instance_idx = np.random.randint(0, len(dist))
            instance, instance_label = dist[instance_idx]
            bag.append(instance.unsqueeze(0).float())

            # Keep track of instance classes
            instance_targets.append(instance_label)
            bag_target = cls.get_bag_target_from_instance_targets(instance_targets)

        bag = torch.stack(bag)
        instance_targets = np.asarray(instance_targets)
        return bag, bag_target, instance_targets

    @staticmethod
    def get_bag_target_from_instance_targets(bag_instance_targets):
        if 1 in bag_instance_targets and 2 in bag_instance_targets:
            return 3
        if 2 in bag_instance_targets:
            return 2
        if 1 in bag_instance_targets:
            return 1
        return 0


class FourMnistBagsDataset(MilDataset):

    name = 'FourMnistBags'
    d_in = 800
    n_expected_dims = 4  # i x c x h x w
    n_classes = 4

    @classmethod
    def create_datasets(cls, mean_bag_size=30, var_bag_size=2, num_train_bags=2500, num_test_bags=1000,
                        random_state=None):
        if random_state is not None:
            np.random.seed(seed=random_state)
        train_mnist_dataset, val_mnist_dataset, test_mnist_dataset = split_mnist_datasets(random_state=random_state)
        train_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_train_bags, train_mnist_dataset)
        val_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, val_mnist_dataset)
        test_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, test_mnist_dataset)
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def _create_dataset(cls, mean_bag_size, var_bag_size, num_bags, original_dataset, discrim_prob=0.1):
        # Split original data into relevant distributions
        # Clz 0 - Non Discrim: N/A    Discrim: 0 to 7
        # Clz 1 - Non Discrim: 0 to 7 Discrim: 8
        # Clz 2 - Non Discrim: 0 to 7 Discrim: 9
        # Clz 3 - Non Discrim: 0 to 7 Discrim: 8, 9

        zero_to_seven_idxs = get_class_idxs(original_dataset, list(range(8)))
        eight_idxs = get_class_idxs(original_dataset, [8])
        nine_idxs = get_class_idxs(original_dataset, [9])
        eight_nine_idxs = get_class_idxs(original_dataset, [8, 9])

        zero_to_seven_data = original_dataset.data[zero_to_seven_idxs]
        eight_data = original_dataset.data[eight_idxs]
        nine_data = original_dataset.data[nine_idxs]
        eight_nine_data = original_dataset.data[eight_nine_idxs]

        zero_to_seven_targets = [0] * len(zero_to_seven_idxs)
        eight_targets = [1] * len(eight_idxs)
        nine_targets = [2] * len(nine_idxs)
        eight_nine_targets = [2 if t == 9 else 1 if t == 8 else 0 for t in original_dataset.targets[eight_nine_idxs]]

        zero_to_seven_dist = list(zip(zero_to_seven_data, zero_to_seven_targets))
        eight_dist = list(zip(eight_data, eight_targets))
        nine_dist = list(zip(nine_data, nine_targets))
        eight_nine_dist = list(zip(eight_nine_data, eight_nine_targets))

        clz_0_dists = (None, zero_to_seven_dist)
        clz_1_dists = (zero_to_seven_dist, eight_dist)
        clz_2_dists = (zero_to_seven_dist, nine_dist)
        clz_3_dists = (zero_to_seven_dist, eight_nine_dist)
        clz_dists = [clz_0_dists, clz_1_dists, clz_2_dists, clz_3_dists]

        clz_weights = [0.25] * 4
        clz_discrim_probas = [1.0] + [discrim_prob] * 3
        bags, targets, instance_targets = cls._create_bags(clz_dists, clz_weights, clz_discrim_probas,
                                                           mean_bag_size, var_bag_size, num_bags)
        return cls(bags, targets, instance_targets)

    @classmethod
    def _create_bags(cls, clz_dists, clz_weights, clz_discrim_probas, mean_bag_size, var_bag_size, num_bags):
        bags = []
        targets = []
        instance_targets = []
        for _ in range(num_bags):
            bag_size = int(np.round(np.random.normal(loc=mean_bag_size, scale=var_bag_size, size=1)))
            if bag_size < 2:
                bag_size = 2

            selected_clz = np.random.choice(range(len(clz_weights)), p=clz_weights)
            non_discrim_dist, discrim_dist = clz_dists[selected_clz]
            discrim_proba = clz_discrim_probas[selected_clz]

            bag, target, bag_instance_targets = cls._create_bag(non_discrim_dist, discrim_dist, discrim_proba, bag_size,
                                                                selected_clz)

            bags.append(bag)
            targets.append(target)
            instance_targets.append(bag_instance_targets)

        return bags, targets, instance_targets

    @classmethod
    def _create_bag(cls, non_discrim_dist, discrim_dist, discrim_proba, bag_size, target):
        bag = []
        instance_targets = []
        bag_target = None

        while len(bag) < bag_size or bag_target is None or bag_target != target:
            if len(bag) == bag_size:
                del bag[0]
                del instance_targets[0]

            # Select if this instance is going to be discriminatory or non-discriminatory
            dist = discrim_dist if np.random.random(1)[0] < discrim_proba else non_discrim_dist

            instance_idx = np.random.randint(0, len(dist))
            instance, instance_label = dist[instance_idx]
            bag.append(instance.unsqueeze(0).float())

            # Keep track of instance classes
            instance_targets.append(instance_label)
            bag_target = cls.get_bag_target_from_instance_targets(instance_targets)

        bag = torch.stack(bag)
        instance_targets = np.asarray(instance_targets)
        return bag, bag_target, instance_targets

    @staticmethod
    def get_bag_target_from_instance_targets(bag_instance_targets):
        if 1 in bag_instance_targets and 2 in bag_instance_targets:
            return 3
        if 2 in bag_instance_targets:
            return 2
        if 1 in bag_instance_targets:
            return 1
        return 0


class CountMnistBagsDataset(MilDataset):

    name = 'CountMnistBags'
    d_in = 800
    n_expected_dims = 4  # i x c x h x w
    n_classes = 1

    @classmethod
    def create_datasets(cls, mean_bag_size=15, var_bag_size=1, num_train_bags=2500, num_test_bags=1000,
                        random_state=None):
        if random_state is not None:
            np.random.seed(seed=random_state)
        train_mnist_dataset, val_mnist_dataset, test_mnist_dataset = split_mnist_datasets(random_state=random_state)
        train_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_train_bags, train_mnist_dataset)
        val_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, val_mnist_dataset)
        test_dataset = cls._create_dataset(mean_bag_size, var_bag_size, num_test_bags, test_mnist_dataset)
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def _create_dataset(cls, mean_bag_size, var_bag_size, num_bags, original_dataset, discrim_prob=0.1):
        # Split original data into relevant distributions
        # Non-Discrim: 0 to 7
        # Discrim -ve: 8
        # Discrim +ve: 9
        zero_to_seven_idxs = get_class_idxs(original_dataset, list(range(8)))
        eight_idxs = get_class_idxs(original_dataset, [8])
        nine_idxs = get_class_idxs(original_dataset, [9])
        # Get data and labels
        zero_to_seven_data = original_dataset.data[zero_to_seven_idxs]
        eight_data = original_dataset.data[eight_idxs]
        nine_data = original_dataset.data[nine_idxs]
        zero_to_seven_targets = [0] * len(zero_to_seven_idxs)
        eight_targets = [-1] * len(eight_idxs)
        nine_targets = [1] * len(nine_idxs)
        zero_to_seven_dist = list(zip(zero_to_seven_data, zero_to_seven_targets))
        eight_dist = list(zip(eight_data, eight_targets))
        nine_dist = list(zip(nine_data, nine_targets))
        # Create bags
        bags, targets, instance_targets = cls._create_bags(zero_to_seven_dist, eight_dist, nine_dist, discrim_prob,
                                                           mean_bag_size, var_bag_size, num_bags)
        # Create dataset
        return cls(bags, targets, instance_targets)

    @classmethod
    def _create_bags(cls, non_discrim_dist, negative_discrim_dst, positive_discrim_dist, discrim_proba,
                     mean_bag_size, var_bag_size, num_bags):
        bags = []
        targets = []
        instance_targets = []
        for _ in range(num_bags):
            # Get bag size
            bag_size = int(np.round(np.random.normal(loc=mean_bag_size, scale=var_bag_size, size=1)))
            if bag_size < 2:
                bag_size = 2
            # Actually create the bag
            bag_data = cls._create_bag(non_discrim_dist, negative_discrim_dst, positive_discrim_dist,
                                       discrim_proba, bag_size)
            bag, target, bag_instance_targets = bag_data
            # Add to our collection!
            bags.append(bag)
            targets.append(target)
            instance_targets.append(bag_instance_targets)
        return bags, targets, instance_targets

    @classmethod
    def _create_bag(cls, non_discrim_dist, negative_discrim_dst, positive_discrim_dist, discrim_proba, bag_size):
        bag = []
        instance_targets = []
        while len(bag) < bag_size:
            # Select if this instance is going to be discriminatory or non-discriminatory
            discrim = np.random.random(1)[0] < discrim_proba
            if discrim:
                dist = positive_discrim_dist if np.random.random(1)[0] < 0.5 else negative_discrim_dst
            else:
                dist = non_discrim_dist
            # Get instance data
            instance_idx = np.random.randint(0, len(dist))
            instance, instance_label = dist[instance_idx]
            # Update bag and instance targets
            bag.append(instance.unsqueeze(0).float())
            instance_targets.append(instance_label)
        # Wrap it all up properly
        bag = torch.stack(bag)
        instance_targets = np.asarray(instance_targets)
        # Work out the bag target from the instance targets
        bag_target = cls.get_bag_target_from_instance_targets(instance_targets)
        return bag, bag_target, instance_targets

    @staticmethod
    def get_bag_target_from_instance_targets(bag_instance_targets):
        # This assumes all instances are correctly labelled as 1 (9), -1 (8), or 0 (0-7)
        return np.sum(bag_instance_targets)
