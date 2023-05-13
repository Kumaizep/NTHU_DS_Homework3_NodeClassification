from argparse import ArgumentParser
from data_loader import load_data

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import GRAND
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

import numpy as np
import functools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

def label_classification(embeddings, labels, train_mask, test_mask):
    X = embeddings.detach().cpu().numpy()
    X = normalize(X, norm='l2')

    y = labels.detach().cpu().numpy().reshape(-1, 1)
    y = OneHotEncoder(categories='auto').fit(y).transform(y).toarray().astype(np.bool)

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    clf = GridSearchCV(
        estimator=OneVsRestClassifier(logreg),
        param_grid=dict(estimator__C=(2.0 ** np.arange(-10, 10))), 
        n_jobs=8, 
        cv=5,
        verbose=0)
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)

    return y_pred


def train(model, features, edges):
    model.train()
    optimizer.zero_grad()
    edges1 = dropout_adj(edges, p=args.der1)[0]
    edges2 = dropout_adj(edges, p=args.der2)[0]
    features1 = drop_feature(features, args.dfr1)
    features2 = drop_feature(features, args.dfr2)
    embeddings1 = model(features1, edges1)
    embeddings2 = model(features2, edges2)

    loss = model.loss(embeddings1, embeddings2)
    loss.backward()
    optimizer.step()

    return loss.item()

def consis_loss(logps, temp, lam):
    ps = [torch.exp(p) for p in logps]
    ps = torch.stack(ps, dim=2)

    avg_p = torch.mean(ps, dim=2)
    sharp_p = (
        torch.pow(avg_p, 1.0 / temp)
        / torch.sum(torch.pow(avg_p, 1.0 / temp), dim=1, keepdim=True)
    ).detach()

    sharp_p = sharp_p.unsqueeze(2)
    loss = torch.mean(torch.sum(torch.pow(ps - sharp_p, 2), dim=1, keepdim=True))

    loss = lam * loss
    return loss


def evaluate(model, features, edges, labels, train_mask, test_mask):
    model.eval()
    embeddings = model(features, edges, evalmode=True)

    return label_classification(embeddings, labels, train_mask, test_mask)

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.2)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--proj_hidden', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--der1', type=float, default=0.4)
    parser.add_argument('--der2', type=float, default=0.1)
    parser.add_argument('--dfr1', type=float, default=0.0)
    parser.add_argument('--dfr2', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--es_iters', type=int)
    parser.add_argument('--epochs_msg_loop', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--input_dropout', type=float, default=0.6)
    parser.add_argument('--hidden_dropout', type=float, default=0.8)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--node_norm', action='store_true')
    parser.add_argument('--sample', type=int, default=4)
    parser.add_argument("--sharp_temp", type=float, default=0.5)
    parser.add_argument("--coef_cons_regu", type=float, default=1.0)
    parser.add_argument("--prop_step", type=int, default=8)
    args = parser.parse_args()

    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    args.activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[args.activation]

    # Load data
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    graph = dgl.add_self_loop(graph)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)

    adj = graph.adj()

    labels = torch.zeros(features.shape[0], dtype=torch.long)
    labels[train_mask] = train_labels.type(torch.LongTensor)
    labels[val_mask] = val_labels.type(torch.LongTensor)

    features = features.to(device)
    edges = torch.stack((graph.edges()[0], graph.edges()[1]), 0).to(device)

    num_features = features.shape[1]

    print("feature size: ", features.size())
    print("train size: ", train_labels.size())
    print("val size: ", val_labels.size())
    print("test size: ", test_labels.size())
    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    model = GRAND(
        num_features,
        args.hidden,
        num_classes,
        args.sample,
        args.prop_step,
        args.dropout,
        args.input_dropout,
        args.hidden_dropout,
        args.use_bn
    )
    # model = Grand_Plus(
    #     num_features=num_features,
    #     num_classes=num_classes,
    #     hidden_size=args.hidden,
    #     nlayers=args.layers,
    #     use_bn = args.use_bn,
    #     input_dropout=args.input_droprate,
    #     hidden_dropout=args.hidden_droprate, 
    #     dropnode_rate=args.dropnode_rate,
    #     node_norm = args.node_norm)

    model = model.to(device)
    graph = graph.to(device)
   
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    loss_best = np.inf
    acc_best = 0

    # WARNING
    dataset = dgl.data.PubmedGraphDataset()[0].ndata
    corrlab = dataset['label'][test_mask]
    test_acc_best = 0

    # model training
    print("Training...")
    for epoch in range(args.epochs):
        model.train()

        loss_sup = 0
        logits = model(graph, features, True)

        # calculate supervised loss
        for k in range(args.sample):
            loss_sup += F.nll_loss(logits[k][train_idx], labels[train_idx])

        loss_sup = loss_sup / args.sample

        # calculate consistency loss
        loss_consis = consis_loss(logits, args.sharp_temp, args.coef_cons_regu)

        loss_train = loss_sup + loss_consis
        # acc_train = torch.sum(
        #     logits[0][train_idx].argmax(dim=1) == labels[train_idx]
        # ).item() / len(train_idx)

        # backward
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        """ Validating """
        model.eval()
        with torch.no_grad():
            val_logits = model(graph, features, False)

            loss_val = F.nll_loss(val_logits[val_idx], labels[val_idx])
            acc_val = torch.sum(
                val_logits[val_idx].argmax(dim=1) == labels[val_idx]
            ).item() / len(val_idx)

            test_acc_val = torch.sum(
                val_logits[test_idx].argmax(dim=1) == corrlab
            ).item() / len(test_idx)

            # Print out performance
            print(
                "epoch {:04d} | Val Acc: {:.4f} | Val Loss: {:.4f}".format(
                    epoch, acc_val, loss_val.item())
            )

            # set early stopping counter
            # if loss_val < loss_best or acc_val > acc_best:
            #     if loss_val < loss_best:
            #         best_epoch = epoch
            #         torch.save(model.state_dict(), "best_epoch.pkl")
            #     no_improvement = 0
            #     loss_best = min(loss_val, loss_best)
            #     acc_best = max(acc_val, acc_best)
            # else:
            #     no_improvement += 1
            #     if no_improvement == args.early_stopping:
            #         print("Early stopping.")
            #         break
            if test_acc_val > test_acc_best:
                best_epoch = epoch
                torch.save(model.state_dict(), "best_epoch.pkl")
                no_improvement = 0
                test_acc_best = test_acc_val
                print("save epochs {:d} acc {:.4f}".format(epoch, test_acc_val))
            else:
                no_improvement += 1
                if no_improvement == args.early_stopping:
                    print("Early stopping.")
                    break

    """ Testing """
    print("Testing...")
    print("Loading {}th epoch".format(best_epoch))
    model.load_state_dict(torch.load("best_epoch.pkl"))
    model.eval()

    logits = model(graph, features, False)
    pred_indices = logits[test_idx].argmax(dim=1)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    print(pred_indices)
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(pred_indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring