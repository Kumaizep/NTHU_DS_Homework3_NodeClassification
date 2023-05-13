# Data Science HW3

## TODO
* model.py
  * implement node classification model
* train.py
  * setup model
* func.py

## Install packages
* scipy, networkx, torch, torch_geometric, torch_scatter, dgl
```
pip install scipy networkx torch torch_geometric torch_scatter dgl
```
* Install dgl
  * https://www.dgl.ai/pages/start.html

* Install pytorch
  * https://pytorch.org/get-started/locally/

## Run sample code
```python
python3 train.py
```
(GPU mode is not recommended as it has not been tested for correct execution. And CPU mode would be fast enough, and the model training can complete in about one minute using AMD Ryzen 4750U.)

## Dataset
* Unknown graph data
  * Label Split:
    * Train: 60, Valid: 500, Test: 1000
* File name description
```
  dataset
  │   ├── private_features.pkl # node feature
  │   ├── private_graph.pkl # graph edges
  │   ├── private_num_classes.pkl # number of classes
  │   ├── private_test_labels.pkl # X
  │   ├── private_test_mask.pkl # nodes indices of testing set
  │   ├── private_train_labels.pkl # nodes labels of training set
  │   ├── private_train_mask.pkl # nodes indices of training set
  │   ├── private_val_labels.pkl # nodes labels of validation set
  │   └── private_val_mask.pkl # nodes indices of validation set
```