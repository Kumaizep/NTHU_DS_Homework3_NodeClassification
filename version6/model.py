import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
from torch_geometric.nn import GCNConv
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.layers = args.layers
        self.input_droprate = args.input_dropout
        self.hidden_droprate = args.hidden_dropout
        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(self.layers):
            size1 = num_features if i == 0 else args.hidden
            size2 = num_classes if i == self.layers - 1 else args.hidden
            self.fcs.append(nn.Linear(size1, size2, bias=True))
            self.bns.append(nn.BatchNorm1d(size1))
        for fc in self.fcs:
            fc.reset_parameters()

    def normalize(self, inputs):
        return inputs / (1e-12 + torch.norm(inputs, p=2, dim=-1, keepdim=True))

    def embs_clac(self, embs, layer, drop):
        embs = self.normalize(embs).detach()
        embs = self.bns[layer](embs)
        embs = F.dropout(embs, drop, training=self.training)
        embs = self.fcs[layer](embs)
        return embs

    def forward(self, embs):
        embs = self.embs_clac(F.relu(embs), 0, self.input_droprate)
        for i in range(1, self.layers):
            embs = self.embs_clac(embs, 0, self.hidden_droprate)

        return embs


class Grand_Plus(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.mlp = MLP(num_features, num_classes, args)
        self.dropout = args.dropout

    def forward(self, X):
        logits = self.mlp(X)
        return logits

    def random_prop(self, feats, err_approx, mat_idx):  
        # print(feats.size())
        # print(mat_scores[:, None].size())
        err_approx = F.dropout(err_approx, p=self.dropout, training=self.training)
        prop_logits = scatter(feats * err_approx[:, None], mat_idx[:, None], dim=0)
        mat_sum_s = scatter(err_approx[:, None], mat_idx[:, None], dim=0)
        return prop_logits / (mat_sum_s + 1e-12)