from overrides import overrides

import pytorch_mil.model.base_models as bm
from pytorch_mil.data.sival.sival_dataset import SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, SIVAL_D_IN
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import modules as mod


def get_model_clz_from_name(model_name):
    if model_name == 'InstanceSpaceNN':
        return SivalInstanceSpaceNN
    if model_name == 'EmbeddingSpaceNN':
        return SivalEmbeddingSpaceNN
    if model_name == 'AttentionNN':
        return SivalAttentionNN
    if model_name == 'MultiHeadAttentionNN':
        return SivalMultiHeadAttentionNN
    if model_name == 'GNN':
        return SivalGNN
    raise ValueError("No SIVAL model class found for model name {:s}".format(model_name))


class SivalEncoder(mod.FullyConnectedStack):

    def __init__(self, dropout):
        super().__init__(SIVAL_D_IN, (128,), 256, dropout, raw_last=False)


class SivalInstanceSpaceNN(bm.InstanceSpaceNN):

    def __init__(self, device, ds_agg_hid=(256, 64), dropout=0.45, agg_func_name='mean'):
        encoder = SivalEncoder(dropout)
        aggregator = agg.InstanceAggregator(encoder.d_out, ds_agg_hid, SIVAL_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }


class SivalEmbeddingSpaceNN(bm.EmbeddedSpaceNN):

    def __init__(self, device, ds_agg_hid=(), dropout=0.25, agg_func_name='max'):
        encoder = SivalEncoder(dropout)
        aggregator = agg.EmbeddingAggregator(encoder.d_out, ds_agg_hid, SIVAL_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-3,
        }


class SivalAttentionNN(bm.AttentionNN):

    def __init__(self, device, ds_agg_hid=(), dropout=0.15, d_attn=256):
        encoder = SivalEncoder(dropout)
        aggregator = agg.AttentionAggregator(encoder.d_out, ds_agg_hid, d_attn, SIVAL_N_CLASSES, dropout)
        super().__init__(device, SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
        }


class SivalMultiHeadAttentionNN(bm.AttentionNN):

    def __init__(self, device, n_heads, ds_agg_hid=(), dropout=0.15, d_attn=256):
        encoder = SivalEncoder(dropout)
        aggregator = agg.MultiHeadAttentionAggregator(n_heads, encoder.d_out, ds_agg_hid,
                                                      d_attn, SIVAL_N_CLASSES, dropout)
        super().__init__(device, SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-5,
        }


class SivalGNN(bm.ClusterGNN):

    def __init__(self, device, d_gnn=64, ds_gnn_hid=(128, 256), ds_fc_hid=(128,), dropout=0.2):
        encoder = SivalEncoder(dropout)
        super().__init__(device, SIVAL_N_CLASSES, SIVAL_N_EXPECTED_DIMS, encoder,
                         encoder.d_out, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-5,
        }
