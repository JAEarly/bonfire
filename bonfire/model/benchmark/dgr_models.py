from overrides import overrides
from torch import nn

from pytorch_mil.data.benchmark.dgr.dgr_dataset import DGR_N_CLASSES, DGR_FV_SIZE, DGR_N_EXPECTED_DIMS
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import models
from pytorch_mil.model import modules as mod


def get_model_clzs():
    return [DgrInstanceSpaceNN]


class DgrEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DGR_FV_SIZE, ds_enc_hid, d_enc, dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.25, agg_func_name='max'):
        encoder = DgrEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, DGR_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, DGR_N_CLASSES, DGR_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-2,
        }
