from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.nn import SAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from pytorch_mil.model import modules as mod, aggregator as agg


def get_model_clz_from_name(model_name):
    if model_name == 'InstanceSpaceNN':
        return InstanceSpaceNN
    if model_name == 'EmbeddingSpaceNN':
        return EmbeddingSpaceNN
    if model_name == 'AttentionNN':
        return AttentionNN
    if model_name == 'MultiHeadAttentionNN':
        return MultiHeadAttentionNN
    if model_name == 'ClusterGNN':
        return ClusterGNN
    raise ValueError("No model class found for model name {:s}".format(model_name))


class MultipleInstanceModel(nn.Module, ABC):

    def __init__(self, device, n_classes, n_expec_dims):
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_expec_dims = n_expec_dims

    @abstractmethod
    def forward(self, model_input):
        pass

    @abstractmethod
    def forward_verbose(self, bags):
        pass


class MultipleInstanceNN(MultipleInstanceModel, ABC):

    def forward(self, model_input):
        # Check if input is unbatched bag (n_expec_dims) or batched (n_expec_dims + 1)
        input_shape = model_input.shape
        unbatched_bag = len(input_shape) == self.n_expec_dims

        # Batch if single bag
        bags = model_input.unsqueeze(0) if unbatched_bag else model_input

        # Actually pass the input through the model
        #  We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(bags)

        # Return single pred if unbatched_bag bag else multiple preds
        if unbatched_bag:
            return bag_predictions.squeeze()
        return bag_predictions


class InstanceEncoder(nn.Module):

    def __init__(self, d_in, ds_hid, d_out, dropout):
        super().__init__()
        self.stack = mod.FullyConnectedStack(d_in, ds_hid, d_out, dropout, raw_last=False)

    def forward(self, x):
        return self.stack(x)

    @staticmethod
    def from_yaml_obj(y):
        return InstanceEncoder(y.d_in, y.ds_hid, y.d_out, y.dropout)


class InstanceSpaceNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_instance_predictions = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, instance_predictions = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_instance_predictions.append(instance_predictions)
        return bag_predictions, bag_instance_predictions

    @staticmethod
    def from_yaml_obj(device, n_classes, n_expec_dims, model_yobj):
        encoder = InstanceEncoder.from_yaml_obj(model_yobj.Encoder)
        aggregator = agg.InstanceAggregator.from_yaml_obj(model_yobj.Aggregator)
        return InstanceSpaceNN(device, n_classes, n_expec_dims, encoder, aggregator)


class EmbeddingSpaceNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, _ = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
        return bag_predictions, None

    @staticmethod
    def from_yaml_obj(device, n_classes, n_expec_dims, model_yobj):
        encoder = InstanceEncoder.from_yaml_obj(model_yobj.Encoder)
        aggregator = agg.EmbeddingAggregator.from_yaml_obj(model_yobj.Aggregator)
        return EmbeddingSpaceNN(device, n_classes, n_expec_dims, encoder, aggregator)


class AttentionNN(MultipleInstanceNN):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_attns = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            bag_prediction, attn = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_attns.append(attn)
        return bag_predictions, bag_attns

    @staticmethod
    def from_yaml_obj(device, n_classes, n_expec_dims, model_yobj):
        encoder = InstanceEncoder.from_yaml_obj(model_yobj.Encoder)
        aggregator = agg.MultiHeadAttentionAggregator.from_yaml_obj(model_yobj.Aggregator)
        return AttentionNN(device, n_classes, n_expec_dims, encoder, aggregator)


# Still need to define this so it matches the YAML file, even though it does exactly the same as AttentionNN
class MultiHeadAttentionNN(AttentionNN):
    pass


class ClusterGNN(MultipleInstanceModel):

    def __init__(self, device, n_classes, n_expec_dims, encoder,
                 d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout):
        super().__init__(device, n_classes, n_expec_dims)
        self.n_clusters = 1
        self.encoder = encoder
        self.gnn_stack = mod.GNNConvStack(d_enc, ds_gnn_hid, d_gnn, dropout, SAGEConv, raw_last=False)
        # TODO this should be called GNN cluster but the trained models are stuck with gnn_pool
        self.gnn_pool = SAGEConv(d_enc, self.n_clusters)
        self.classifier = mod.FullyConnectedStack(d_gnn, ds_fc_hid, n_classes, dropout, raw_last=True)

    def forward(self, model_input):
        # Unbatched input could be Data type, or raw tensor without Data structure
        unbatched_bag = type(model_input) is Data or \
                        (type(model_input) is torch.Tensor and len(model_input.shape) == self.n_expec_dims)

        # Batch if single bag
        bags = [model_input] if unbatched_bag else model_input

        # Actually pass the input through the model
        #  We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(bags)

        # Return single pred if unbatched_bag bag else multiple preds
        if unbatched_bag:
            return bag_predictions.squeeze()
        return bag_predictions

    def forward_verbose(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_cluster_weights = []
        for i, bag in enumerate(bags):
            # If given Tensor and not Data type, add fully connected graph to data
            if type(bag) is not Data:
                bag = self.add_graph(bag)
            bag = bag.to(self.device)
            x, edge_index = bag.x, bag.edge_index
    
            # Embed instances
            instance_embeddings = self.encoder(x)

            # Run instance embeddings through GNN stack for message passing
            updated_instance_embeddings, _ = self.gnn_stack((instance_embeddings, edge_index))

            # Clustering
            cluster_weights = F.softmax(self.gnn_pool(instance_embeddings, edge_index), dim=0)

            # Reduce to single representation of the graph
            graph_embedding, _, _, _ = dense_diff_pool(updated_instance_embeddings, to_dense_adj(edge_index),
                                                       cluster_weights)
            graph_embedding = graph_embedding.squeeze(dim=0)

            # Classify
            bag_prediction = self.classifier(graph_embedding)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_cluster_weights.append(cluster_weights.T)
        return bag_predictions, bag_cluster_weights

    @staticmethod
    def add_graph(bag):
        edge_index = torch.ones((len(bag), len(bag)))
        edge_index, _ = dense_to_sparse(edge_index)
        edge_index = edge_index.long().contiguous()
        graph_bag = Data(x=bag, edge_index=edge_index)
        return graph_bag

    @staticmethod
    def from_yaml_obj(device, n_classes, n_expec_dims, model_yobj):
        encoder = InstanceEncoder.from_yaml_obj(model_yobj.Encoder)
        y = model_yobj.GNN
        return ClusterGNN(device, n_classes, n_expec_dims, encoder, y.d_enc, y.d_gnn, y.ds_gnn_hid, y.ds_fc_hid,
                          model_yobj.TrainParams.dropout)
