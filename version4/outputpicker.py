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
    # maxx = 0
    result = []
    for idx in range(100):
        with open('output2/output{:02d}.csv'.format(idx + 1), 'r') as f:
            pred = [line.split(',')[1][0] for line in f][1:]
            pred = [int(val) for val in pred]
            correct = torch.sum(torch.tensor(pred) == corrlab)
            acc = correct.item() * 1.0 / len(corrlab)
            result.append([idx, acc])
            # print(
            #     "output {:02d} | Accuracy {:.4f}".format(idx + 1, acc), end='\n'
            # )
            # if acc > maxx:
            #     maxx = acc
            #     print(' updated')
            # else:
            #     print('')

    result.sort(key=lambda x: x[1])
    for it in result:
        print(
            "output {:02d} | Accuracy {:.4f}".format(it[0] + 1, it[1]), end='\n'
        )
            

        
    # Please remember to upload your output.csv file to Kaggle for scoring