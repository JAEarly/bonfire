import torch
from overrides import overrides
from torch import nn

from bonfire.data.benchmark.dgr.dgr_dataset import DgrDataset
from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod


def get_model_clzs():
    return [DgrInstanceSpaceNN, DgrEmbeddingSpaceNN]


class DgrEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DgrDataset.d_in, ds_enc_hid, d_enc, dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.25, agg_func_name='mean'):
        encoder = DgrEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, DgrDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, DgrDataset.n_classes, DgrDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-5,
            'weight_decay': 0.01,
        }


class DgrEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.25, agg_func_name='mean'):
        encoder = DgrEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, DgrDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, DgrDataset.n_classes, DgrDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-4,
        }
