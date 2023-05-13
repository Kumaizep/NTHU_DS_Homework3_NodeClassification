import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv

# def optimize(params, lr=0.01):
#     if run == 0:
#         print('params:', sum(p.numel() for p in params))
#     return optim.Adam(params, lr=lr)


# def speye(n):
#     return torch.sparse_coo_tensor(
#         torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)


# def spnorm(A, eps=1e-5):
#     D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -1
#     indices = A._indices()
#     return gpu(torch.sparse_coo_tensor(indices, D[indices[0]], size=A.size()))


# def FC(din, dout):
#     return gpu(nn.Sequential(
#         nn.BatchNorm1d(din),
#         nn.LayerNorm(din),
#         nn.LeakyReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(din, dout)))


# class MLP(nn.Module):
#     def __init__(self, din, hid, dout, n_layers=3, A=None):
#         super(self.__class__, self).__init__()
#         self.A = A
#         self.layers = nn.ModuleList()
#         self.layers.append(gpu(nn.Linear(din, hid)))
#         for _ in range(n_layers - 2):
#             self.layers.append(FC(hid, hid))
#         self.layers.append(FC(hid, dout))

#     def forward(self, x):
#         for layer in self.layers:
#             if self.A is not None:
#                 x = self.A @ x
#             x = layer(x)
#         return x

# GCN = MLP
