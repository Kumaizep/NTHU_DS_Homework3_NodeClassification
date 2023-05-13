from argparse import ArgumentParser

from data_loader import load_data

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# from model import GCN
# from model import YourGNNModel # Build your model in model.py
    
import os
import warnings
warnings.filterwarnings("ignore")

gpu = lambda x: x
if torch.cuda.is_available():
    print("Using Cuda")
    dev = torch.device('cuda')
    gpu = lambda x: x.to(dev)

def evaluate(logits, labels, mask):
    """Evaluate model accuracy"""
    with torch.no_grad():
        indices = logits[mask].max(dim=1).indices
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def pred(logits, mask):
    return logits[mask].max(dim=1).indices

# def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, es_iters=None):
    
#     # define train/val samples, loss function and optimizer
#     loss_fcn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

#     # If early stopping criteria, initialize relevant parameters
#     if es_iters:
#         print("Early stopping monitoring on")
#         loss_min = 1e8
#         es_i = 0

#     # training loop
#     for epoch in range(epochs):
#         model.train()
#         logits = model(g, features)
#         loss = loss_fcn(logits[train_mask], train_labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         acc = evaluate(g, features, val_labels, val_mask, model)
#         print(
#             "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
#                 epoch, loss.item(), acc
#             )
#         )
        
#         val_loss = loss_fcn(logits[val_mask], val_labels).item()
#         if es_iters:
#             if val_loss < loss_min:
#                 loss_min = val_loss
#                 es_i = 0
#             else:
#                 es_i += 1

#             if es_i >= es_iters:
#                 print(f"Early stopping at epoch={epoch+1}")
#                 break


def optimize(params, lr=0.01):
    if run == 0:
        print('params:', sum(p.numel() for p in params))
    return optim.Adam(params, lr=lr)


def speye(n):
    return torch.sparse_coo_tensor(
        torch.arange(n).view(1, -1).repeat(2, 1), [1] * n)


def spnorm(A, eps=1e-5):
    D = (torch.sparse.sum(A, dim=1).to_dense() + eps) ** -1
    indices = A._indices()
    return gpu(torch.sparse_coo_tensor(indices, D[indices[0]], size=A.size()))


def FC(din, dout):
    return gpu(nn.Sequential(
        nn.BatchNorm1d(din),
        nn.LayerNorm(din),
        nn.LeakyReLU(),
        nn.Dropout(0.5),
        nn.Linear(din, dout)))


class MLP(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3, A=None):
        super(self.__class__, self).__init__()
        self.A = A
        self.layers = nn.ModuleList()
        self.layers.append(gpu(nn.Linear(din, hid)))
        for _ in range(n_layers - 2):
            self.layers.append(FC(hid, hid))
        self.layers.append(FC(hid, dout))

    def forward(self, x):
        for layer in self.layers:
            if self.A is not None:
                x = self.A @ x
            x = layer(x)
        return x

GCN = MLP

class LinkDist(nn.Module):
    def __init__(self, din, hid, dout, n_layers=3):
        super(self.__class__, self).__init__()
        self.mlp = MLP(din, hid, hid, n_layers=n_layers - 1)
        self.out = FC(hid, dout)
        self.inf = FC(hid, dout)

    def forward(self, x):
        x = self.mlp(x)
        return self.out(x), self.inf(x)

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    node_num = features.shape[0]
    labels = torch.zeros(node_num, dtype=torch.long)
    labels[train_mask] = train_labels.type(torch.LongTensor)
    labels[val_mask] = val_labels.type(torch.LongTensor)

    g_method = 'colinkdist-trans'
    g_data = 'pubmed?nthu'
    g_split = '0'

    g_split = float(g_split)
    epochs = args.epochs
    batch_size = 1024
    hid = args.hidden
    n_layers = args.layers

    X = node_features = gpu(features)
    Y = node_labels = gpu(labels)
    n_nodes = node_num
    nrange = torch.arange(n_nodes)
    n_features = node_features.shape[1]
    n_labels = int(Y.max().item() + 1)
    src, dst = graph.edges()
    n_edges = src.shape[0]
    is_bidir = ((dst == src[0]) & (src == dst[0])).any().item()
    print('BiDirection: %s' % is_bidir)
    print('nodes: %d' % n_nodes)
    print('features: %d' % n_features)
    print('classes: %d' % n_labels)
    print('edges: %d' % (
        (n_edges - (src == dst).sum().item()) / (1 + is_bidir)))
    degree = n_edges * (2 - is_bidir) / n_nodes
    print('degree: %.2f' % degree)
    
    # model training
    print("Training...")
    # train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, args.es_iters)
  
    for run in range(1):
        torch.manual_seed(run)
        valid_mask = val_mask
        
        train_idx = nrange[train_mask]
        known_idx = nrange[~(valid_mask | test_mask)]
        E = speye(n_nodes)

        if '-trans' in g_method:
            A = [spnorm(graph.adj() + E, eps=0)] * 2
        else:
            # Inductive Settings
            src, dst = graph.edges()
            flt = ~(
                valid_mask[src] | test_mask[src]
                | valid_mask[dst] | test_mask[dst])
            src = src[flt]
            dst = dst[flt]
            n_edges = src.shape[0]
            A = torch.sparse_coo_tensor(
                torch.cat((
                    torch.cat((src, dst), dim=0).unsqueeze(0),
                    torch.cat((dst, src), dim=0).unsqueeze(0)), dim=0),
                values=torch.ones(2 * n_edges),
                size=(n_nodes, n_nodes))
            A = (spnorm(A + E), spnorm(graph.adj() + E, eps=0))

        A = spnorm(graph.adj())
        
        linkdist = LinkDist(n_features, hid, n_labels, n_layers=n_layers)
        opt = optimize([*linkdist.parameters()])
        if '-trans' in g_method:
            src, dst = graph.edges()
            n_edges = src.shape[0]
        # Ratio of known labels in nodes
        train_nprob = train_mask.sum().item() / n_nodes
        # Ratio of known labels in edges
        train_eprob = ((
            train_mask[src].sum() + train_mask[dst].sum()
        ) / (2 * n_edges)).item()
        # Hyperparameter alpha
        alpha = 1 - train_eprob
        label_ndist = Y[
            torch.arange(n_nodes)[train_mask]].float().histc(n_labels)
        label_edist = (
            Y[src[train_mask[src]]].float().histc(n_labels)
            + Y[dst[train_mask[dst]]].float().histc(n_labels))
        # label_edist = label_edist + 1
        weight = n_labels * F.normalize(
            label_ndist / label_edist, p=1, dim=0)
        # print("------------------------------------")
        for epoch in range(0, 1 + int(epochs // degree)):
            # print("------------------------------------")
            linkdist.train()
            # Hyperparameter beta
            if g_split:
                beta = 0.1
                beta1 = beta * train_nprob / (train_nprob + train_eprob)
                beta2 = beta - beta1
            else:
                beta1 = train_nprob
                beta2 = train_eprob
            idx = torch.randint(0, n_nodes, (n_edges, ))
            smax = lambda x: torch.softmax(x, dim=-1)
            for perm in DataLoader(range(n_edges), batch_size=batch_size, shuffle=True):
                opt.zero_grad()
                pidx = idx[perm]
                psrc = src[perm]
                pdst = dst[perm]
                y, z = linkdist(X[pidx])
                y1, z1 = linkdist(X[psrc])
                y2, z2 = linkdist(X[pdst])
                loss = alpha * (
                    F.mse_loss(y1, z2) + F.mse_loss(y2, z1)
                    - 0.5 * (
                        F.mse_loss(smax(y1), smax(z))
                        + F.mse_loss(smax(y2), smax(z))
                        + F.mse_loss(smax(y), smax(z1))
                        + F.mse_loss(smax(y), smax(z2))
                    )
                )
                m = train_mask[psrc]
                if m.any().item():
                    target = Y[psrc][m]
                    loss = loss + (
                        F.cross_entropy(y1[m], target, weight=weight)
                        + F.cross_entropy(z2[m], target, weight=weight)
                        - beta1 * F.cross_entropy(
                            z[m], target, weight=weight))
                m = train_mask[pdst]
                if m.any().item():
                    target = Y[pdst][m]
                    loss = loss + (
                        F.cross_entropy(y2[m], target, weight=weight)
                        + F.cross_entropy(z1[m], target, weight=weight)
                        - beta1 * F.cross_entropy(
                            z[m], target, weight=weight))
                m = train_mask[pidx]
                if m.any().item():
                    target = Y[pidx][m]
                    loss = loss + (2 * F.cross_entropy(y[m], target) - beta2 * 
                        (F.cross_entropy(z1[m], target) + F.cross_entropy(z2[m], target)))
                # print("Loss {:.4f} ".format(loss), end=' ')
                loss.backward()
                opt.step()
            with torch.no_grad():
                linkdist.eval()
                Z, S = linkdist(X)
                acc = evaluate(F.log_softmax(Z, dim=-1) + alpha * (A @ F.log_softmax(S, dim=-1)), labels[val_mask], val_mask)
                print(
                    "\nEpoch {:05d} | Accuracy {:.4f} ".format(
                        epoch, acc
                    )
                )
    
  
    print("Testing...")
    with torch.no_grad():
        linkdist.eval()
        Z, S = linkdist(X)
        indices = pred(F.log_softmax(Z, dim=-1) + alpha * (A @ F.log_softmax(S, dim=-1)), val_mask)
        # print(logits)
        # _, indices = torch.max(logits, dim=1)
        # print(torch.max(logits, dim=1))
        print(indices)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring