import os

import cv2
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torchvision import transforms
from tqdm import tqdm

from bonfire.data.mil_dataset import MilDataset
from bonfire.train.metrics import RegressionMetric


def load_metadata_df():
    metadata_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'metadata.csv'))
    metadata_df = metadata_df[metadata_df['split'] == 'train']
    metadata_df = metadata_df[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df['sat_image_path'] = metadata_df['sat_image_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    metadata_df['mask_path'] = metadata_df['mask_path'].apply(
        lambda img_pth: os.path.join(RAW_DATA_DIR, img_pth))
    return metadata_df


def load_class_dict_df():
    class_dict_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'class_dict.csv'))
    class_dict_df[['r', 'g', 'b']] = class_dict_df[['r', 'g', 'b']].apply(lambda v: v // 255)
    class_dict_df['target'] = list(range(len(class_dict_df)))
    return class_dict_df


RAW_DATA_DIR = 'data/DGR/raw'
TARGET_OUT_PATH = 'data/DGR/targets.csv'
COVER_DIST_PATH = 'data/DGR/cover_dist.csv'
METADATA_DF = load_metadata_df()
CLASS_DICT_DF = load_class_dict_df()

basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.2811, 0.3786, 0.4077), (0.0696, 0.0759, 0.1054))])


class DGRDataset(MilDataset):

    d_in = 1200
    n_expected_dims = 4  # i x c x h x w
    n_classes = 2  # multi-regression with two outputs (urban land, agriculture land)

    def __init__(self, bags, targets, bags_metadata):
        super().__init__(bags, targets, None, bags_metadata)
        self.transform = basic_transform

    @classmethod
    def create_datasets(cls, random_state=12, grid_size=153, patch_size=28):
        bags, targets, bags_metadata = DGRDataset.load_dgr_bags(patch_size=patch_size)

        train_bags, test_bags, train_targets, test_targets, train_bags_metadata, test_bags_metadata = \
            train_test_split(bags, targets, bags_metadata, train_size=0.6, random_state=random_state)
        val_bags, test_bags, val_targets, test_targets, val_bags_metadata, test_bags_metadata = \
            train_test_split(test_bags, test_targets, test_bags_metadata, train_size=0.5, random_state=random_state)

        train_dataset = DGRDataset(train_bags, train_targets, train_bags_metadata)
        val_dataset = DGRDataset(val_bags, val_targets, val_bags_metadata)
        test_dataset = DGRDataset(test_bags, test_targets, test_bags_metadata)

        return train_dataset, val_dataset, test_dataset

    def create_complete_dataset(cls):
        raise NotImplementedError

    def __getitem__(self, index):
        instances = self._load_instances(index)
        target = self.targets[index]
        return instances, target

    def _load_instances(self, bag_idx):
        instances = []
        bag = self.bags[bag_idx]
        for patch_path in bag:
            instance = cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                instance = self.transform(instance)
            instances.append(instance)
        instances = torch.stack(instances)
        return instances

    @staticmethod
    def load_dgr_bags(grid_size=153, patch_size=28):
        patches_df = pd.read_csv("data/DGR/patch_{:d}_{:d}_data.csv".format(grid_size, patch_size))
        coverage_df = DGRDataset.load_per_class_coverage()
        complete_df = pd.merge(patches_df, coverage_df, on='image_id')
        bags = [s.split(",") for s in complete_df['patch_paths'].tolist()]
        targets = complete_df[['urban_land', 'agriculture_land']].to_numpy()
        bags_metadata = [{'id': id_} for id_ in complete_df['image_id'].tolist()]
        return bags, targets, bags_metadata

    @staticmethod
    def target_to_rgb(target):
        r = CLASS_DICT_DF.loc[CLASS_DICT_DF['target'] == target]
        rgb = r[['r', 'g', 'b']].values.tolist()[0]
        return rgb

    @staticmethod
    def target_to_name(target):
        return CLASS_DICT_DF['name'][target]

    @staticmethod
    def make_mask_binary(mask):
        binary_mask = torch.zeros_like(torch.as_tensor(mask))
        binary_mask[mask > 128] = 1
        return binary_mask

    @staticmethod
    def visualise_data():
        for i in range(len(METADATA_DF)):
            sat_path = METADATA_DF['sat_image_path'][i]
            mask_path = METADATA_DF['mask_path'][i]
            sat_img = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
            mask_img = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            mask_img = DGRDataset.make_mask_binary(mask_img)

            fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(14, 4))
            axes[0][0].imshow(sat_img, vmin=0, vmax=255)
            axes[0][1].imshow(mask_img.float(), vmin=0, vmax=1)
            for target in range(7):
                single_mask = DGRDataset.mask_for_single_target(mask_img, target)
                axes[1][target].imshow(single_mask, vmin=0, vmax=1, cmap='gray')
                axes[1][target].set_title(DGRDataset.target_to_name(target))
            plt.tight_layout()
            plt.show()

    @staticmethod
    def load_per_class_coverage():
        if not os.path.exists(COVER_DIST_PATH):
            DGRDataset.generate_per_class_coverage()
        return pd.read_csv(COVER_DIST_PATH)

    @staticmethod
    def generate_per_class_coverage():
        cover_dist_df = METADATA_DF[['image_id']].copy()
        for name in CLASS_DICT_DF['name']:
            cover_dist_df[name] = pd.Series(dtype=float)

        for i in tqdm(range(len(METADATA_DF)), 'Calculating class coverage for each image'):
            mask_path = METADATA_DF['mask_path'][i]
            mask_img = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
            mask_img = DGRDataset.make_mask_binary(mask_img)

            s = 0
            for target in range(7):
                single_mask = DGRDataset.mask_for_single_target(mask_img, target)
                name = DGRDataset.target_to_name(target)
                percentage_cover = len(single_mask.nonzero())/single_mask.numel()
                cover_dist_df.loc[i, name] = percentage_cover
                s += percentage_cover
            assert abs(s - 1) < 1e-6

        cover_dist_df.to_csv(COVER_DIST_PATH, index=False)

    @staticmethod
    def plot_per_class_coverage():
        cover_dist_df = DGRDataset.load_per_class_coverage()

        fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(14, 2))
        for target in range(7):
            name = DGRDataset.target_to_name(target)
            dist = cover_dist_df[name]
            axes[target].set_title(DGRDataset.target_to_name(target))
            axes[target].hist(dist, bins=25, range=(0, 1))
            axes[target].set_xlabel('Coverage')
            axes[target].set_ylabel('Density')
            axes[target].set_ylim(0, 1000)
        plt.tight_layout()
        fig_path = "data/DGR/class_coverage_histograms.png"
        fig.savefig(fig_path, format='png', dpi=300)
        plt.show()

    @staticmethod
    def mask_for_single_target(mask_img, target):
        rgb = DGRDataset.target_to_rgb(target)
        c1 = mask_img[:, :, 0] == rgb[0]
        c2 = mask_img[:, :, 1] == rgb[1]
        c3 = mask_img[:, :, 2] == rgb[2]
        new_mask = (c1 & c2 & c3).int()
        return new_mask

    @staticmethod
    def extract_grid_patches(grid_size=153, patch_size=28):
        num_patches = int(2448 / grid_size * 2448 / grid_size)
        print('{:d} patches per image'.format(num_patches))
        reconstructed_dim = int(num_patches ** 0.5 * patch_size)
        print('{:d} x {:d} effective new size'.format(reconstructed_dim, reconstructed_dim))

        patch_dir = 'data/DGR/patch_{:d}_{:d}'.format(grid_size, patch_size)
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)

        patches_df = METADATA_DF[['image_id']].copy()
        patches_df['patch_paths'] = ""

        for i in tqdm(range(len(METADATA_DF)), desc='Extracting patches'):
            image_id = METADATA_DF['image_id'][i]
            sat_path = METADATA_DF['sat_image_path'][i]
            sat_img = cv2.imread(sat_path)  # BGR

            n_x = int(sat_img.shape[0]/grid_size)
            n_y = int(sat_img.shape[1]/grid_size)

            patch_paths = []
            for i_x in range(n_x):
                for i_y in range(n_y):
                    p_x = i_x * grid_size
                    p_y = i_y * grid_size
                    patch_img = sat_img[p_x:p_x+grid_size, p_y:p_y+grid_size, :]
                    patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                    patch_path = "{:s}/{:d}_{:d}_{:d}.png".format(patch_dir, image_id, i_x, i_y)
                    patch_paths.append(patch_path)

                    cv2.imwrite(patch_path, patch_img)  # BGR

            patches_df.loc[i, 'patch_paths'] = ",".join(patch_paths)

        patches_df.to_csv("data/DGR/patch_{:d}_{:d}_data.csv".format(grid_size, patch_size), index=False)

    @classmethod
    def baseline_performance(cls):
        train_dataset, val_dataset, test_dataset = cls.create_datasets()
        train_mean_target = train_dataset.targets.mean()

        def performance_for_dataset(pred, dataset):
            targets = dataset.targets
            preds = torch.ones_like(targets)
            preds *= pred
            metric = RegressionMetric.calculate_metric(preds, targets, None)
            print('MSE Loss: {:.4f}'.format(metric.mse_loss))
            print('MAE Loss: {:.4f}'.format(metric.mae_loss))

        print('-- Train --')
        performance_for_dataset(train_mean_target, train_dataset)
        print('-- Val --')
        performance_for_dataset(train_mean_target, val_dataset)
        print('-- Test --')
        performance_for_dataset(train_mean_target, test_dataset)

    @classmethod
    def calculate_dataset_normalisation(cls):
        bags, _, _ = cls.load_dgr_bags()
        avgs = []
        transformation = transforms.ToTensor()
        for bag in tqdm(bags, "Calculating dataset norm"):
            for file_name in bag:
                img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
                avg = torch.mean(transformation(img), dim=(1, 2))
                avgs.append(avg)
        arrs = torch.stack(avgs)
        print(arrs.shape)
        arrs_mean = torch.mean(arrs, dim=0)
        arrs_std = torch.std(arrs, dim=0)
        print(arrs_mean)
        print(arrs_std)


if __name__ == "__main__":
    DGRDataset.extract_grid_patches()
