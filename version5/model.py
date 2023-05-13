import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
from torch_geometric.nn import GCNConv
from torch_scatter import scatter


def drop_node(features, drop_rate, training):
    n = features.shape[0]
    # print(type(np.ones(n)))
    # print(type(drop_rate))
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)

    if training:
        masks = torch.bernoulli(1.0 - drop_rates).unsqueeze(1)
        features = masks.to(features.device) * features

    else:
        features = features * (1.0 - drop_rate)

    return features


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_drop, hidden_drop, use_bn=False):
        super().__init__()
        self.layer1 = nn.Linear(nfeat, nhid, bias=True)
        self.layer2 = nn.Linear(nhid, nclass, bias=True)
        self.input_dropout = nn.Dropout(input_drop)
        self.hidden_dropout = nn.Dropout(hidden_drop)
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        if self.use_bn:
            x = self.bn1(x)
        x = self.input_dropout(x)
        x = F.relu(self.layer1(x))

        if self.use_bn:
            x = self.bn2(x)
        x = self.hidden_dropout(x)
        x = self.layer2(x)

        return x


def GRANDConv(graph, features, pror_step):
    with graph.local_scope():
        # Calculate Symmetric normalized adjacency matrix 
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(features.device).unsqueeze(1)

        graph.ndata["norm"] = norm
        graph.apply_edges(fn.u_mul_v("norm", "norm", "weight"))

        x = features
        y = 0 + features

        for i in range(pror_step):
            graph.ndata["h"] = x
            graph.update_all(fn.u_mul_e("h", "weight", "m"), fn.sum("m", "h"))
            x = graph.ndata.pop("h")
            y.add_(x)

    return y / (pror_step + 1)


class GRAND(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, sapmle=1, prop_step=3, dropout=0.0, input_droprate=0.0,
        hidden_droprate=0.0, batchnorm=False):
        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.sapmle = sapmle
        self.prop_step = prop_step
        self.num_classes = num_classes

        self.mlp = MLP(
            in_dim, hid_dim, num_classes, input_droprate, hidden_droprate, batchnorm
        )

        self.dropout = dropout
        # self.dropout = nn.Dropout(dropout)

    def forward(self, graph, features, training=True):
        X = features
        sapmle = self.sapmle

        if training:  # Training Mode
            output_list = []
            for s in range(sapmle):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.prop_step)  # Graph Convolution
                output_list.append(
                    torch.log_softmax(self.mlp(feat), dim=-1)
                )  # Prediction

            return output_list
        else:  # Inference Mode
            drop_feat = drop_node(X, self.dropout, False)
            X = GRANDConv(graph, drop_feat, self.prop_step)

            return torch.log_softmax(self.mlp(X), dim=-1)