from overrides import overrides
from torch import nn

from bonfire.data.benchmark.mnist.mnist_bags import FourMnistBagsDataset
from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod


def get_model_clzs():
    return [FourMnistInstanceSpaceNN, FourMnistEmbeddingSpaceNN, FourMnistAttentionNN, FourMnistGNN, FourMnistMiLstm]


class MnistEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=1, c_out=20, kernel_size=5, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=20, c_out=50, kernel_size=5, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(FourMnistBagsDataset.d_in, ds_enc_hid, d_enc, dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FourMnistInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(), ds_agg_hid=(128, 64,), dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, FourMnistBagsDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }


class FourMnistEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(128,), ds_agg_hid=(), dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, FourMnistBagsDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-3,
        }


class FourMnistAttentionNN(models.AttentionNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.15, d_attn=64):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, d_enc, ds_agg_hid, d_attn,
                                                      FourMnistBagsDataset.n_classes, dropout)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-4,
        }


class FourMnistGNN(models.ClusterGNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), d_gnn=128, ds_gnn_hid=(128, 128), ds_fc_hid=(64,),
                 dropout=0.3):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-5,
            'weight_decay': 1e-5,
        }


class FourMnistMiLstm(models.MiLstm):

    def __init__(self, device, d_enc=128, ds_enc_hid=(), d_lstm_hid=128, n_lstm_layers=1,
                 bidirectional=False, ds_fc_hid=(), dropout=0.2):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(d_enc, d_lstm_hid, n_lstm_layers, bidirectional,
                                                      dropout, ds_fc_hid, FourMnistBagsDataset.n_classes)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-5,
            'weight_decay': 1e-8,
        }
