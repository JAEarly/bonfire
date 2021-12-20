from overrides import overrides

from pytorch_mil.data.musk_dataset import MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, MUSK_D_IN
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import models
from pytorch_mil.model import modules as mod


class MuskInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=512, ds_enc_hid=(), ds_agg_hid=(256, 64), dropout=0.45, agg_func_name='mean'):
        encoder = mod.FullyConnectedStack(MUSK_D_IN, ds_enc_hid, d_enc, dropout, raw_last=False)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, MUSK_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }


class MuskEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(128,), ds_agg_hid=(), dropout=0.25, agg_func_name='max'):
        encoder = mod.FullyConnectedStack(MUSK_D_IN, ds_enc_hid, d_enc, dropout, raw_last=False)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, MUSK_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-3,
        }


class MuskAttentionNN(models.AttentionNN):

    def __init__(self, device, d_enc=128, ds_enc_hid=(128, 256), ds_agg_hid=(), dropout=0.15, d_attn=256):
        encoder = mod.FullyConnectedStack(MUSK_D_IN, ds_enc_hid, d_enc, dropout, raw_last=False)
        aggregator = agg.MultiHeadAttentionAggregator(1, d_enc, ds_agg_hid, d_attn, MUSK_N_CLASSES, dropout)
        super().__init__(device, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
        }


class MuskGNN(models.ClusterGNN):

    def __init__(self, device,
                 d_enc=128, ds_enc_hid=(), d_gnn=64, ds_gnn_hid=(128, 256), ds_fc_hid=(128,), dropout=0.2):
        encoder = mod.FullyConnectedStack(MUSK_D_IN, ds_enc_hid, d_enc, dropout, raw_last=False)
        super().__init__(device, MUSK_N_CLASSES, MUSK_N_EXPECTED_DIMS, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }