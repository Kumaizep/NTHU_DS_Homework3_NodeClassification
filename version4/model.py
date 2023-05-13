import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from dgl.nn import GraphConv
from torch_geometric.nn import GCNConv

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation, layers= 2):
        super().__init__()
        self.layers = layers
        self.conv = [GCNConv(in_channels, 2 * out_channels)]
        for _ in range(1, layers - 1):
            self.conv.append(GCNConv(2 * out_channels, 2 * out_channels))
        self.conv.append(GCNConv(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation

    def forward(self, features: torch.Tensor, edges: torch.Tensor):
        for i in range(self.layers):
            features = self.activation(self.conv[i](features, edges))
        return features

class Model(nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau):
        super().__init__()
        self.encoder = encoder
        self.tau = tau
        self.linear1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.linear2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, features, edges, evalmode=False):
        if evalmode:
            self.encoder.eval()
        else:
            self.encoder.train()
        return self.encoder(features, edges)

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(
            torch.mm(F.normalize(z1), F.normalize(z1).t())
            )
        between_sim = f(
            torch.mm(F.normalize(z1), F.normalize(z2).t())
            )

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, embeddings1, embeddings2):
        z1 = self.linear2(F.elu(self.linear1(embeddings1)))
        z2 = self.linear2(F.elu(self.linear1(embeddings2)))

        l1 = self.semi_loss(z1, z2)
        l2 = self.semi_loss(z2, z1)

        ret = (l1 + l2) * 0.5
        ret = ret.mean()

        return ret

def drop_feature(features, drop_prob):
    drop_mask = torch.empty((features.size(1),), dtype=torch.float32, device=features.device).uniform_(0, 1) < drop_prob
    features = features.clone()
    features[:, drop_mask] = 0

    return features