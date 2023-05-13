from argparse import ArgumentParser

from data_loader import load_data

import torch
import torch.nn as nn

from model import GCN, GRACE, data_augmentation
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


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics

        return wrapper

    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    reti = np.zeros(y_pred.shape, np.int)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
        reti[i][0] = indices[i]
    return reti


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()


# @repeat(3)
def label_classification(embeddings, y, train_mask, test_mask, split='public', ratio=0.1):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    if split == 'random':
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=1 - ratio)
    elif split == 'public':
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = Y[train_mask]
        y_test = Y[test_mask]

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    return y_pred

def evaluate(graph, features, train_labels, val_labels, train_mask, val_mask, model):
    """Evaluate model accuracy"""
    model.eval()
    graphp = graph.add_self_loop().to(device)
    featuresp = features.to(device)
    embeds = model.get_embedding(graphp, featuresp)
    labels = torch.zeros(19717, dtype=torch.int)
    labels[train_mask] = train_labels.type(torch.IntTensor)
    labels[val_mask] = val_labels.type(torch.IntTensor)

    indices = label_classification(embeds, labels, train_mask, val_mask)

    indices = torch.tensor([idx[0] for idx in indices])
    correct = torch.sum(indices == val_labels)
    return correct.item() * 1.0 / len(val_labels)

def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, device, 
    drop_feature_rate_1=0.2, drop_feature_rate_2=0.2, drop_edge_rate_1=0.2, drop_edge_rate_2=0.2, es_iters=None):
    
    # define train/val samples, loss function and optimizer
    # loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # If early stopping criteria, initialize relevant parameters
    if es_iters:
        print("Early stopping monitoring on")
        loss_min = 1e8
        es_i = 0

    # training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        graph1, features1 = data_augmentation(graph, features, drop_feature_rate_1, drop_edge_rate_1)
        graph2, features2 = data_augmentation(graph, features, drop_feature_rate_2, drop_edge_rate_2)

        # logits = model(g, features)
        # loss = loss_fcn(logits[train_mask], train_labels)
        loss = model(graph1.to(device), graph2.to(device), features1.to(device), features2.to(device))
        loss.backward()
        optimizer.step()

        acc = evaluate(g, features, train_labels, val_labels, train_mask, val_mask, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )
        # print(
        #     "Epoch {:05d} | Loss {:.4f}".format(
        #         epoch, loss.item()
        #     )
        # )
        
        # val_loss = loss_fcn(logits[val_mask], val_labels).item()
        # if es_iters:
        #     if val_loss < loss_min:
        #         loss_min = val_loss
        #         es_i = 0
        #     else:
        #         es_i += 1

        #     if es_i >= es_iters:
        #         print(f"Early stopping at epoch={epoch+1}")
        #         break


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
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

    print(features.size())
    print(train_labels.size())
    print(val_labels.size())
    print(test_labels.size())
    print(train_mask.size())
    print(val_mask.size())
    print(test_mask.size())
    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    in_size = features.shape[1]
    hid_size = 16
    out_size = num_classes
    layer_num = 2
    model = GRACE(in_size, hid_size, out_size, layer_num, nn.ReLU(), 1.0).to(device)
    
    # model training
    print("Training...")
    train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, 
        device, 0.2, 0.2, 0.2, 0.2, args.es_iters)
    
    print("Testing...")
    graph = graph.add_self_loop().to(device)
    features = features.to(device)
    embeds = model.get_embedding(graph, features)
    labels = torch.zeros(19717, dtype=torch.int)
    labels[train_mask] = train_labels.type(torch.IntTensor)
    labels[val_mask] = val_labels.type(torch.IntTensor)

    indices = label_classification(embeds, labels, train_mask, test_mask)


    # model.eval()
    # with torch.no_grad():
    #     logits = model(graph, features)
    #     logits = logits[test_mask]
    #     _, indices = torch.max(logits, dim=1)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring