import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GraphConv

def drop_feature(features, drop_prob):
    drop_mask = th.empty((features.size(1),), dtype=th.float32, device=features.device).uniform_(0, 1) < drop_prob
    features = features.clone()
    features[:, drop_mask] = 0

    return features

def mask_edge(graph, mask_prob):
    mask_rates = th.FloatTensor(np.ones(graph.num_edges()) * mask_prob)
    masks = th.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

def data_augmentation(graph, features, feat_drop_rate, edge_mask_rate):
    edge_mask = mask_edge(graph, edge_mask_rate)
    feat = drop_feature(features, feat_drop_rate)

    ng = dgl.graph(
        (graph.edges()[0][edge_mask], graph.edges()[1][edge_mask]), 
        num_nodes=graph.num_nodes()
        ).add_self_loop()
    return ng, feat

class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size, activation, num_layers = 2):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        self.convs.append(GraphConv(in_size, hid_size))
        for i in range(self.num_layers - 2):
            self.convs.append(GraphConv(hid_size, hid_size))

        self.convs.append(GraphConv(hid_size, out_size))
        self.activation = activation

    def forward(self, graph, feat):
        for i in range(self.num_layers):
            feat = self.activation(self.convs[i](graph, feat))

        return feat

class MLP(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layer1 = nn.Linear(in_size, out_size)
        self.layer2 = nn.Linear(out_size, in_size)

    def forward(self, x):
        return self.layer2(F.elu(self.layer1(x)))


class GRACE(nn.Module):
    """
    TODO: Use GCN model as reference, implement your own model here to achieve higher accuracy on testing data
    """
    def __init__(self, in_size, hid_size, out_size, num_layers, activation, tao):
        super().__init__()
        self.encoder = GCN(in_size, hid_size * 2, hid_size, activation, num_layers)
        self.proj = MLP(hid_size, out_size)
        self.tao = tao

    def norm_multi(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return th.mm(z1, z2.t())

    def get_loss(self, z1, z2):
        intra_view_pairs = th.exp(self.norm_multi(z1, z1) / self.tao)
        inter_view_pairs = th.exp(self.norm_multi(z1, z2) / self.tao)

        deno = inter_view_pairs.sum(1) + intra_view_pairs.sum(1) - intra_view_pairs.diag()
        loss = -th.log(inter_view_pairs.diag() / deno)

        return loss

    def get_embedding(self, graph, features):
        h = self.encoder(graph, features)

        return h.detach()

    def forward(self, graph1, graph2, features1, features2):
        # encoding
        h1 = self.encoder(graph1, features1)
        h2 = self.encoder(graph2, features2)

        # projection
        z1 = self.proj(h1)
        z2 = self.proj(h2)

        # get loss
        l1 = self.get_loss(z1, z2)
        l2 = self.get_loss(z2, z1)

        ret = (l1 + l2) * 0.5

        return ret.mean()

