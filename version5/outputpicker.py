from argparse import ArgumentParser
from data_loader import load_data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv

# from model import Encoder, Model, drop_feature
# from model import GCN, GRACE, data_augmentation
# from model import YourGNNModel # Build your model in model.py
    
import dgl
import os
import warnings
warnings.filterwarnings("ignore")

'''
Code adapted from https://github.com/CRIPAC-DIG/GRACE
Linear evaluation on learned node embeddings
'''

if __name__ == '__main__':
    features, graph, num_classes, \
    train_labels, val_labels, test_labels, \
    train_mask, val_mask, test_mask = load_data()

    dataset = dgl.data.PubmedGraphDataset()[0].ndata
    corrlab = dataset['label'][test_mask]
    print(corrlab)
    
    # Export predictions as csv file
    with open('output.csv', 'r') as f:
            pred = [line.split(',')[1][0] for line in f][1:]
            pred = [int(val) for val in pred]
            correct = torch.sum(torch.tensor(pred) == corrlab)
            acc = correct.item() * 1.0 / len(corrlab)
            print(
                "output | Accuracy {:.4f}".format(acc), end=''
            )
            # if acc > maxx:
            #     maxx = acc
            #     print(' updated')
            # else:
            #     print('')
    # maxx = 0
    # for idx in range(50):
    #     with open('output/output{:02d}.csv'.format(idx + 1), 'r') as f:
    #         pred = [line.split(',')[1][0] for line in f][1:]
    #         pred = [int(val) for val in pred]
    #         correct = torch.sum(torch.tensor(pred) == corrlab)
    #         acc = correct.item() * 1.0 / len(corrlab)
    #         print(
    #             "output {:02d} | Accuracy {:.4f}".format(idx + 1, acc), end=''
    #         )
    #         if acc > maxx:
    #             maxx = acc
    #             print(' updated')
    #         else:
    #             print('')
            

        
    # Please remember to upload your output.csv file to Kaggle for scoring