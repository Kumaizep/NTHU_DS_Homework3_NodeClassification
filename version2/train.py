from argparse import ArgumentParser

from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import GCN
from func import normalize_tensor, eval_acc, train_model, evaluate
# from model import YourGNNModel # Build your model in model.py
    
import os
import warnings
warnings.filterwarnings("ignore")

# def evaluate(g, features, labels, mask, model):
#     """Evaluate model accuracy"""
#     model.eval()
#     with torch.no_grad():
#         logits = model(g, features)
#         logits = logits[mask]
#         _, indices = torch.max(logits, dim=1)
#         correct = torch.sum(indices == labels)
#         return correct.item() * 1.0 / len(labels)

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


if __name__ == '__main__':

    parser = ArgumentParser()
    # you can add your arguments if needed
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--layers', type=int, default=1)
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


    # Preliminaries
    node_num = features.shape[0]
    labels = torch.zeros(node_num, dtype=torch.long)
    labels[train_mask] = train_labels.type(torch.LongTensor)
    labels[val_mask] = val_labels.type(torch.LongTensor)

    adj_mat = graph.adj()
    adj_low = normalize_tensor(torch.eye(node_num) + adj_mat)
    adj_high = (torch.eye(node_num) - adj_low).to(device).to_sparse()
    adj_low = adj_low.to(device)
    adj_mat = None
    
    # deg_mat = torch.zeros(adj_mat.size(), dtype=torch.int)
    # for idx, deg in enumerate(graph.in_degrees()):
    #     deg_mat[idx] = deg

    # lap_mat = deg_mat - adj_mat

    # eig_val, eig_vec = torch.linalg.eig(lap_mat)

    # sym_lap_mat = 

    criterion = nn.NLLLoss()
    eval_func = eval_acc

    # t_total = time.time() 
    epoch_total = 0
    result = np.zeros(1)
    
    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=num_classes,
        nlayers=args.layers,
        nnodes=features.shape[0],
        dropout=args.dropout,
        structure_info=0,
    ).to(device)

    idx_train, idx_val, idx_test = train_mask, val_mask, test_mask
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-3)

    # idx_train = idx_train.cuda()
    # idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()
    # model.cuda()

    curr_res = 0
    curr_training_loss = None
    best_val_loss = float("inf")
    val_loss_history = torch.zeros(args.epochs)

    print("Training...")
    test_preds = test_labels
    for epoch in range(args.epochs):
        # t = time.time()
        acc_train, loss_train = train_model(
            model,
            optimizer,
            adj_low,
            adj_high,
            adj_mat,
            features,
            labels,
            idx_train,
            criterion,
        )

        model.eval()
        output = model(features, adj_low, adj_high, adj_mat)
        output = F.log_softmax(output, dim=1)
        val_loss, val_acc = criterion(output[idx_val], labels[idx_val]), evaluate(
            output, labels, idx_val, eval_func
        )
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, val_loss, val_acc
            )
        )
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            curr_res = evaluate(output, labels, idx_test, eval_func)
            test_preds = output[idx_test]
            curr_training_loss = loss_train

        if epoch >= 0:
            val_loss_history[epoch] = val_loss.detach()


    epoch_total = epoch_total + epoch

    # Testing
    # result[idx] = curr_res
    del model, optimizer
    # if args.cuda:
    #     torch.cuda.empty_cache()



    
    # Initialize the model (Baseline Model: GCN)
    """TODO: build your own model in model.py and replace GCN() with your model"""
    # in_size = features.shape[1]
    # out_size = num_classes
    # model = GCN(in_size, 16, out_size).to(device)
    # print(out_size)
    
    # # model training
    # print("Training...")
    # train(graph, features, train_labels, val_labels, train_mask, val_mask, model, args.epochs, args.es_iters)
    
    # print("Testing...")
    # model.eval()
    # with torch.no_grad():
    #     logits = model(graph, features)
    #     logits = logits[test_mask]
    #     # print(logits)
    #     _, indices = torch.max(logits, dim=1)
    #     # print(torch.max(logits, dim=1))
    #     # print(indices)
    
    # Export predictions as csv file
    print("Export predictions as csv file.")
    indices = test_preds.max(1)[1].type_as(labels)
    print(indices)
    with open('output.csv', 'w') as f:
        f.write('Id,Predict\n')
        for idx, pred in enumerate(indices):
            f.write(f'{idx},{int(pred)}\n')
    # Please remember to upload your output.csv file to Kaggle for scoring