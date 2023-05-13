import multiprocessing as mp
import numpy as np
import torch

def unpack_topk_adj(topk_adj, features, index):
    aim_topk_adj = topk_adj[index]
    source_idx, neighbor_idx = aim_topk_adj.nonzero()
    source_idx = torch.tensor(source_idx, dtype=torch.long)
    aim_feat = features[neighbor_idx]
    err_approx = torch.tensor(aim_topk_adj.data[aim_topk_adj.data.nonzero()], dtype=torch.float32)
    return source_idx, aim_feat, err_approx


def get_local_logits(device, model, feats):
    logits = []
    with torch.set_grad_enabled(False):
        for i in range(0, feats.shape[0], 10000):
            batch_attr = torch.FloatTensor(feats[i:i + 10000]).to(device)
            logits.append(model(batch_attr).to(device).numpy())
    logits = np.row_stack(logits)
    return logits

def get_perturbed_features(order, adj, features_np):
    pfeats = features_np.copy()
    deg_row_inv = 1 / np.maximum(adj.sum(1), 1e-12)
    for _ in range(order - 1):
        features_np =  np.multiply(deg_row_inv[:,None], (adj.dot(features_np)))
        pfeats += features_np
    pfeats = pfeats / order
    return pfeats

def iterate_batch(index, batch_size, shuffle=False):
    numSamples = len(index)
    sub_idx = np.arange(numSamples)
    if shuffle:
        np.random.shuffle(sub_idx)
    for start_idx in range(0, numSamples, batch_size):
        end_idx = min(start_idx + batch_size, numSamples)
        picked_idx = sub_idx[start_idx:end_idx]
        yield index[picked_idx]

def sample_unlabel(sample_idx, unlabel_batch_size):
    sub_idx = np.arange(sample_idx.shape[0])
    np.random.shuffle(sub_idx)
    picked_idx = sub_idx[:unlabel_batch_size]
    return sample_idx[picked_idx]

def clac_loss_con(unlabled_preds, tau, gamma):
    unlabled_preds = [torch.exp(it) for it in unlabled_preds]
    sum_pred_prob = 0.0
    for pred_prob in unlabled_preds:
        sum_pred_prob += pred_prob
    avg_pred_prob = sum_pred_prob / len(unlabled_preds)

    sharp_avg_pred_prob = torch.pow(avg_pred_prob, 1./tau)
    pseudo_labels = (sharp_avg_pred_prob / torch.sum(sharp_avg_pred_prob, dim=1, keepdim=True)).detach()

    loss = 0.0
    for pred_prob in unlabled_preds:
        dist = (pred_prob - pseudo_labels).pow(2).sum(1).sqrt()
        loss += sum(dist[avg_pred_prob.max(1)[0] > gamma]) / len(pred_prob)
    loss = loss / len(unlabled_preds)
    return loss

def augm_assign(aim_dict, key, val):
    if not key in aim_dict:
        aim_dict[key] = 0.0
    aim_dict[key] += val
    return aim_dict

def sub_matrix_approx(degree, indptr, indices, node_id, coef, rmax, top_k):
    residue_value = {}
    reserve_value = {}

    residue_value[node_id] = 1.0
    reserve_value[node_id] = 0.0
    for i in range(coef.size):
        residue_value_tmp = {}
        while residue_value:
            oldNode = list(residue_value.items())[0][0]
            r = residue_value.pop(oldNode, None)
            reserve_value = augm_assign(reserve_value, oldNode, coef[i] * r)
            if i != coef.size - 1:
                if degree[oldNode] == 0:
                    residue_value_tmp = augm_assign(residue_value_tmp, node_id, r)
                elif r >= rmax * degree[oldNode]:
                    for j in range(indptr[oldNode], indptr[oldNode + 1]):
                        residue_value_tmp = augm_assign(residue_value_tmp, indices[j], r / degree[oldNode])
        residue_value = residue_value_tmp

    res = np.array([[int(key), val] for key, val in reserve_value.items()])
    res_sec = np.array([x[1] for x in res])
    
    k = min(top_k, int(res.size/2))
    nth_elemed_idx = np.argpartition(res_sec, - k)
    res = res[nth_elemed_idx[-k:]]

    return [res, k]

def GFPush(train_unlabel_idx, coef, rmax, top_k, indptr, indices):
    degree = [0] * (indptr.size - 1)
    for i in range(indptr.size - 1):
        degree[i] = indptr[i+1] - indptr[i]

    row_idx   = np.zeros((train_unlabel_idx.shape[0] * top_k), dtype=np.int32)
    col_idx   = np.zeros((train_unlabel_idx.shape[0] * top_k), dtype=np.int32)
    mat_value = np.zeros((train_unlabel_idx.shape[0] * top_k), dtype=np.float64)

    inputs = []
    for it in range(train_unlabel_idx.size):
        inputs.append((degree, indptr, indices, train_unlabel_idx[it], coef, rmax, top_k))
    pool = mp.Pool(64)
    result = pool.starmap(sub_matrix_approx, inputs)

    for it in range(train_unlabel_idx.size):
        k = result[it][1]
        res = result[it][0]
        for i in range(k):
            idx = it * top_k + i
            if res[i][1] > 0.0:
                row_idx[idx] = train_unlabel_idx[it]
                col_idx[idx] = int(res[i][0])
                mat_value[idx] = res[i][1]

    return row_idx, col_idx, mat_value