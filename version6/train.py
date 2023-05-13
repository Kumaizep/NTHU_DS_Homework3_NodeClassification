from argparse import ArgumentParser
from data_loader import load_data

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import networkx as nx
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Grand_Plus
from func import *
# from model import Encoder, Model, drop_feature
# from model import GCN, GRACE, data_augmentation
# from model import YourGNNModel # Build your model in model.py
    
import os
import warnings
warnings.filterwarnings("ignore")

'''
Code adapted from https://github.com/CRIPAC-DIG/GRACE
Linear evaluation on learned node embeddings
'''

def get_valid_info(model, topk_adj, features, val_idx, labels, batch_size):
    model.eval()

    outputs = []
    for val_index in iterate_batch(val_idx, batch_size):
        source_idx, valid_feat, err_approx = unpack_topk_adj(topk_adj, features, val_index)
        with torch.no_grad():
            valid_feat_aug = model.random_prop(valid_feat, err_approx, source_idx).detach()
            output = model(valid_feat_aug)
            output = torch.log_softmax(output, dim=-1)
        outputs.append(output)
    outputs = torch.cat(outputs)

    loss     = F.nll_loss(outputs, labels[val_idx])
    preds    = outputs.max(1)[1].type_as(labels)
    correct  = torch.sum(torch.tensor(preds) == labels[val_idx])
    acc_rate = correct / len(val_idx)
    return loss, acc_rate

def predict(args, adj, features_np, model, idx_test):
    model.eval()
    
    perturbed_features = get_perturbed_features(args.order, adj, features_np)
    preds = get_local_logits(args.device, model.mlp, perturbed_features).argmax(1)

    print(preds[idx_test])
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(preds[idx_test]):
            f.write(f'{idx},{int(pred)}\n')

def get_argument():
    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=72)
    parser.add_argument('--proj_hidden', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--der1', type=float, default=0.4)
    parser.add_argument('--der2', type=float, default=0.1)
    parser.add_argument('--dfr1', type=float, default=0.0)
    parser.add_argument('--dfr2', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--es_iters', type=int)
    parser.add_argument('--epochs_msg_loop', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0.2)
    parser.add_argument('--hidden_dropout', type=float, default=0.2)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--augmented_times', type=int, default=2)
    parser.add_argument("--sharp_tau", type=float, default=0.5)
    parser.add_argument("--lambda_max", type=float, default=1.0)
    parser.add_argument("--prop_step", type=int, default=8)
    parser.add_argument("--unlabel_num", type=int, default=560)
    parser.add_argument("--seed1", type=int, default=42)
    parser.add_argument("--seed2", type=int, default=0)
    parser.add_argument("--eval_batch", type=int, default=10)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--clip_norm", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--rmax", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--unlabel_batch_size", type=int, default=100)
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--sample_offset", type=int, default=1000)
    args = parser.parse_args()
    args.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]

    if args.use_gpu:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        args.device = torch.device("cpu")

    return args

if __name__ == '__main__':

    print("Loading arguments and graph data ... ", end='')
    args = get_argument()
    np.random.seed(args.seed2)

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    print("Done")
    print("==========================================================================================================")

    print("Set up part1 - Initialize process variable of graph data ... ", end='')
    num_vertex = features.shape[0]
    num_features = features.shape[1]

    train_idx = train_mask.nonzero().squeeze().to(args.device)
    val_idx   = val_mask.nonzero().squeeze().to(args.device)
    test_idx  = test_mask.nonzero().squeeze().to(args.device)
    ul_idx    = test_idx + torch.ones(test_idx.size(), dtype=torch.long) * args.sample_offset

    adj = nx.adjacency_matrix(dgl.to_networkx(graph))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(num_vertex)

    labels = torch.zeros(num_vertex, dtype=torch.long)
    labels[train_mask] = train_labels.type(torch.LongTensor)
    labels[val_mask]   = val_labels.type(torch.LongTensor)

    features_np = features.numpy()
    features    = features.to(args.device)
    print("Done")
    print("==========================================================================================================")

    print("Set up part2 - Generate approximate matrix by GFPush ... ", end='')
    sample_idx        = np.random.permutation(ul_idx)[:args.unlabel_num]
    unlabel_idx       = np.concatenate([val_idx, sample_idx])
    train_unlabel_idx = np.concatenate([train_idx, unlabel_idx])
    coef    = np.ones(args.order, dtype=np.float64) / (args.order)
    indptr  = np.array(adj.indptr, dtype=np.int32)
    indices = np.array(adj.indices, dtype=np.int32)
    row_idx, col_idx, mat_value = GFPush(train_unlabel_idx, coef, args.rmax, args.top_k, indptr, indices)
    print("Done")
    print("==========================================================================================================")

    print("Set up part3 - Construct Grand+ model ... ", end='')
    topk_adj = sp.coo_matrix((mat_value, (row_idx, col_idx)), shape=(num_vertex, num_vertex))
    topk_adj = topk_adj.tocsr()

    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    model = Grand_Plus(num_features, num_classes, args)

    model = model.to(args.device)
    labels = labels.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    print("Done")
    print("==========================================================================================================")
    
    bad_counter = 0
    loss_min    = np.inf
    acc_max     = 0.0
    best_epoch  = 0
    num_batch   = 0

    # model training
    print("Training...")
    for epoch in range(args.epochs):
        for train_index in iterate_batch(train_idx, args.batch_size, shuffle=True):
            model.train()
            optimizer.zero_grad()

            # Get index of union of L_t and U_t for this betch
            unlabel_index_batch = sample_unlabel(sample_idx, args.unlabel_batch_size)
            batch_index = np.concatenate((train_index, unlabel_index_batch))

            # Get index, features, and error-bounded approximation of this betch
            source_idx, batch_feat, err_approx = unpack_topk_adj(topk_adj, features, batch_index)

            unlabled_preds = []
            loss_sup = 0.0
            for i in range(args.augmented_times):
                batch_feat_aug = model.random_prop(batch_feat, err_approx, source_idx).detach()
                batch_pred = model(batch_feat_aug).log_softmax(dim=-1)
                unlabled_preds.append(batch_pred[len(train_index):])
                loss_sup += F.nll_loss(batch_pred[:len(train_index)], labels[train_index])       
            loss_sup = loss_sup / args.augmented_times
            lambda_t = args.lambda_max * min(1, float(num_batch) / args.warmup)
            loss_con = clac_loss_con(unlabled_preds, args.sharp_tau, 2.0 / num_classes)
            loss_train = loss_sup + lambda_t * loss_con

            loss_train.backward()
            clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            if num_batch % args.eval_batch == 0:
                loss_val, acc_val = get_valid_info(model, topk_adj, features, val_idx, labels, args.batch_size)
                print('epoch {:05d} | batch {:05d} | Loss {:.4f} | Accuracy {:.4f}'.format(
                    epoch, num_batch, loss_val, acc_val))
                if acc_val >= acc_max:
                    if loss_val<= loss_min:
                        bad_counter = 0
                        loss_min    = loss_val
                        acc_max     = acc_val
                        best_epoch  = epoch
                        torch.save(model.state_dict(),f"best_epoch.pkl")
                else:
                    bad_counter += 1
                if bad_counter >= args.patience:
                    print('Early stop\n')
                    break
            num_batch += 1
        if bad_counter >= args.patience:
            break

    print('Train result: Min loss {:.4f} | Max accuracy {:.4f}'.format(loss_min, acc_max))
    print("==========================================================================================================")


    """ Testing """
    print("Testing...")
    print("Loading the model status of {}th epoch".format(best_epoch))
    model.load_state_dict(torch.load(f"best_epoch.pkl"))
    predict(args, adj, features_np, model, test_idx)
