import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

from dgl.nn.pytorch import GraphConv

device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        nnodes,
        output_layer=0,
        variant=False,
    ):
        super(GraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.output_layer = output_layer
        self.variant = variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
            Parameter(torch.FloatTensor(in_features, out_features).to(device)),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
            Parameter(torch.FloatTensor(1 * out_features, 1).to(device)),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1).to(device)
        )
        self.struc_low = Parameter(torch.FloatTensor(nnodes, out_features).to(device))
        self.att_vec = Parameter(torch.FloatTensor(3, 3).to(device))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention4(self, output_low, output_high, output_mlp, struc_low):
        T = 4
        feature_concat = torch.cat(
            [
                torch.mm((output_low), self.att_vec_low),
                torch.mm((output_high), self.att_vec_high),
                torch.mm((output_mlp), self.att_vec_mlp),
                torch.mm((struc_low), self.att_struc_low),
            ],
            1,
        )

        logits = torch.mm(torch.sigmoid(feature_concat), self.att_vec) / T

        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None], att[:, 3][:, None]

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        output = 0
        if self.variant:

            output_low = torch.spmm(
                adj_low, F.relu(torch.mm(input, self.weight_low))
            )

            output_high = torch.spmm(
                adj_high, F.relu(torch.mm(input, self.weight_high))
            )
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        else:
            output_low = F.relu(
                torch.spmm(adj_low, (torch.mm(input, self.weight_low)))
            )
            output_high = F.relu(
                torch.spmm(adj_high, (torch.mm(input, self.weight_high)))
            )
            output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return 3 * (
            self.att_low * output_low
            + self.att_high * output_high
            + self.att_mlp * output_mlp
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class MLP(nn.Module):
    """adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py"""

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5
    ):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
            self.bns.append(nn.BatchNorm1d(out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph["node_feat"]
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x


class GCN(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        nnodes,
        dropout,
        structure_info,
        variant=False,
    ):
        super(GCN, self).__init__()

        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.structure_info, self.nlayers, self.nnodes = (
            structure_info,
            nlayers,
            nnodes,
        )

        self.gcns.append(
            GraphConvolution(
                nfeat,
                nhid,
                nnodes,
                variant=variant,
            )
        )
        self.gcns.append(
            GraphConvolution(
                1 * nhid,
                nclass,
                nnodes,
                output_layer=1,
                variant=variant,
            )
        )

        self.dropout = dropout
        self.fea_param, self.xX_param = Parameter(
            torch.FloatTensor(1, 1).to(device)
        ), Parameter(torch.FloatTensor(1, 1).to(device))


    def forward(self, x, adj_low, adj_high, adj_mat):
        x = F.dropout(x, self.dropout, training=self.training)

        feat = self.gcns[0](x, adj_low, adj_high, adj_mat)

        feat = F.dropout((F.relu(feat)), self.dropout, training=self.training)
            
        feat = self.gcns[1](feat, adj_low, adj_high, adj_mat)

        return feat
