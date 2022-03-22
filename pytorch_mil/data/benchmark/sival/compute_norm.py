import torch

from pytorch_mil.data.benchmark.sival.sival_dataset import parse_data_from_file

# TODO move into dataset file

_, bags, _, _ = parse_data_from_file()
print(bags[0].shape)

bags = torch.cat(bags)

print(bags.shape)

arrs_mean = torch.mean(bags, dim=0)
arrs_std = torch.std(bags, dim=0)

print(arrs_mean)
print(arrs_std)
