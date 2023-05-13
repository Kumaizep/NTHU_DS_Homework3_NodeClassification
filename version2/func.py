import torch
import torch.nn.functional as F

def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

@torch.no_grad()
def evaluate(output, labels, split_idx, eval_func):
    acc = eval_func(labels[split_idx], output[split_idx])
    return acc

def eval_acc(y_true, y_pred):
    if y_true.dim() > 1:
        acc_list = []
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        return sum(acc_list) / len(acc_list)
    else:
        preds = y_pred.max(1)[1].type_as(y_true)
        correct = preds.eq(y_true).double()
        correct = correct.sum()
        return correct / len(y_true)

def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def train_model(
    model,
    optimizer,
    adj_low,
    adj_high,
    adj_low_unnormalized,
    features,
    labels,
    idx_train,
    criterion,
):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_low, adj_high, adj_low_unnormalized)
    output = F.log_softmax(output, dim=1)
    # print(type(output[0]))
    # print(type(labels[0]))
    # output = output.type(torch.LongTensor) 
    # labels = labels.type(torch.LongTensor) 
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(labels[idx_train], output[idx_train])

    loss_train.backward()
    optimizer.step()

    return 100 * acc_train.item(), loss_train.item()