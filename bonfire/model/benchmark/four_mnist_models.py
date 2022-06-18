from torch import nn

from bonfire.data.benchmark.mnist.mnist_bags import FourMnistBagsDataset
from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod


def get_model_clzs():
    return [FourMnistInstanceSpaceNN, FourMnistEmbeddingSpaceNN, FourMnistAttentionNN, FourMnistGNN,
            FourMnistEmbeddingSpaceLSTM]


MNIST_D_ENC = 128
MNIST_DS_ENC_HID = (512,)
MNIST_DS_AGG_HID = (64,)


class MnistEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=1, c_out=20, kernel_size=5, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=20, c_out=50, kernel_size=5, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(FourMnistBagsDataset.d_in, MNIST_DS_ENC_HID, MNIST_D_ENC,
                                                dropout, raw_last=False)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FourMnistInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(dropout)
        aggregator = agg.InstanceAggregator(MNIST_D_ENC, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes,
                                            dropout, agg_func_name)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, dropout=0.3, agg_func_name='mean'):
        encoder = MnistEncoder(dropout)
        aggregator = agg.EmbeddingAggregator(MNIST_D_ENC, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes,
                                             dropout, agg_func_name)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistAttentionNN(models.AttentionNN):

    def __init__(self, device, dropout=0.15, d_attn=64):
        encoder = MnistEncoder(dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, MNIST_D_ENC, MNIST_DS_AGG_HID, d_attn,
                                                      FourMnistBagsDataset.n_classes, dropout)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistGNN(models.ClusterGNN):

    def __init__(self, device, dropout=0.3):
        encoder = MnistEncoder(dropout)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims, encoder,
                         MNIST_D_ENC, MNIST_D_ENC, (MNIST_D_ENC,), MNIST_DS_AGG_HID, dropout)


class FourMnistEmbeddingSpaceLSTM(models.MiLstm):

    def __init__(self, device, bidirectional=False, dropout=0.2):
        encoder = MnistEncoder(dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(MNIST_D_ENC, (MNIST_D_ENC,), 1, bidirectional,
                                                      dropout, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)
