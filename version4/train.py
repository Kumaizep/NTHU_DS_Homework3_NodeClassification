from argparse import ArgumentParser
from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

from model import Encoder, Model, drop_feature
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


def evaluate(model, features, edges, labels, train_mask, test_mask):
    model.eval()
    embeddings = model(features, edges, evalmode=True)

    return label_classification(embeddings, labels, train_mask, test_mask)

if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--proj_hidden', type=int, default=256)
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--der1', type=float, default=0.4)
    parser.add_argument('--der2', type=float, default=0.1)
    parser.add_argument('--dfr1', type=float, default=0.0)
    parser.add_argument('--dfr2', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--weight_decay', type=float, default=0.00001)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--es_iters', type=int, help='num of iters to trigger early stopping')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--epochs_msg_loop', type=int, default=1)
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

    labels = torch.zeros(features.shape[0], dtype=torch.long)
    labels[train_mask] = train_labels.type(torch.LongTensor)
    labels[val_mask] = val_labels.type(torch.LongTensor)

    features = features.to(device)
    edges = torch.stack((graph.edges()[0], graph.edges()[1]), 0).to(device)

    num_features = features.shape[1]

    print(features.size())
    print(train_labels.size())
    print(val_labels.size())
    print(test_labels.size())
    print(train_mask.size())
    print(val_mask.size())
    print(test_mask.size())
    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    encoder = Encoder(num_features, args.hidden, args.activation, args.layers).to(device)
    model = Model(encoder, args.hidden, args.proj_hidden, args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # model training
    print("Training...")
    for epoch in range(args.epochs):
        loss = train(model, features, edges)     
        # if epoch % args.epochs_msg_loop == args.epochs_msg_loop - 1:
        #     pred_indices = evaluate(model, features, edges, labels, train_mask, val_mask)
        #     correct = torch.sum(torch.tensor(pred_indices) == val_labels)
        #     acc = correct.item() * 1.0 / len(val_labels)
        #     print(
        #         "Epoch {:04d} | Loss {:.4f} | Accuracy {:.4f}".format(epoch + 1, loss, acc)
        #     )
        # else:
        #     print(
        #         "Epoch {:04d} | Loss {:.4f}".format(epoch + 1, loss)
        #     )
        print(
            "Epoch {:04d} | Loss {:.4f}".format(epoch + 1, loss)
        )

        # if epoch >= args.epochs - 200:
        #     pred_indices = evaluate(model, features, edges, labels, train_mask | val_mask, test_mask)
        #     with open('output{:02d}.csv'.format(args.epochs - epoch), 'w') as f:
        #         f.write('Id,Predict\n')
        #         for idx, pred in enumerate(pred_indices):
        #             f.write(f'{idx},{int(pred)}\n')
    
    print("Testing...")
    pred_indices = evaluate(model, features, edges, labels, train_mask | val_mask, test_mask)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    print(pred_indices)
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(pred_indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring