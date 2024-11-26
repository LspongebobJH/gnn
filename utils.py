import os
import pickle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import random
import os

from torchmetrics import Accuracy, AUROC, AveragePrecision, MeanSquaredError
from sklearn.model_selection import train_test_split, KFold
from torch_geometric.data import Batch
from torch_geometric.transforms import SIGN
from torch_scatter import scatter
import torch.nn as nn

from torch_geometric.data import Batch, Data
from time import time

device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
SINGLE_MODALITY_MODELS = ['GCN', 'SAGE', 'SGC', 'GAT', 'GIN', 'Transformer']
FUSE_SINGLE_MODALITY_MODELS = \
    [name + '_fuse_embed' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_graph' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_pred' for name in SINGLE_MODALITY_MODELS]
FUSE_SINGLE_MODALITY_MODELS_NOSIA = \
    [name + '_fuse_embed_nosia' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_graph_nosia' for name in SINGLE_MODALITY_MODELS] + \
    [name + '_fuse_pred_nosia' for name in SINGLE_MODALITY_MODELS]

def model_infer(model, model_name, **kwargs):
    """
    adjs: adj matrices
    idx: split index 
    raw_Xs: original Xs
    data_lisr: list type of pyg data
    device
    """
    if model_name == 'MHGCN':
        adjs, idx, raw_Xs = kwargs['adjs'], kwargs['idx'], kwargs['raw_Xs']
        logits = model(adjs[idx], raw_Xs[idx])
        
    elif model_name in ['NeuroPath', 'Mew'] + SINGLE_MODALITY_MODELS + \
            FUSE_SINGLE_MODALITY_MODELS + \
            FUSE_SINGLE_MODALITY_MODELS_NOSIA:
        data = kwargs['data']
        logits = model(data)

    return logits.squeeze()

def load_dataset(label_type='classification', eval_type='split', split_args: dict = None, cross_args: dict = None):
    """

    label_type: if classification, all labels(int) are converted into its index. If regression, use original values.
        note: even if it's a regression task in nature, if labels are int, sometimes classification loss function,
        such as cross-entropy loss, has better performance.
    eval_type: if eval_type == split, then train-valid-test split style evaluation. elif eval_type == cross, then n_fold
                cross evalidation
    """
    # TODO train-valid-test split, and cross-validation
    assert label_type in ['classification', 'regression']
    assert eval_type in ['split', 'cross']
    if eval_type == 'split':
        assert split_args is not None

    file_path = f'./dataset/processed_data_{label_type}_{eval_type}.pkl'
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        adjs = data['adjs']
        raw_Xs = data['raw_Xs']
        labels = data['labels']
        mu_lbls = data['mu_lbls']
        std_lbls = data['std_lbls']
    else:
        # data_path = '/home/jiahang/gnn/dataset'
        data_path = './dataset'
        path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
        with open(path, 'rb') as f:
            data_FC = pickle.load(f)

        path = os.path.join(data_path, 'SC.pkl')
        with open(path, 'rb') as f:
            data_SC = pickle.load(f)

        path = os.path.join(data_path, 'T1.pkl')
        with open(path, 'rb') as f:
            data_raw_X = pickle.load(f)

        path = os.path.join(data_path, 'demo.pkl')
        with open(path, 'rb') as f:
            data_labels = pickle.load(f)
        
        # NOTE note that only data_SC has full keys, is it a semi-supervised task?
        ## we only take graph which have labels and both SC, FC modalities.
        adjs = torch.zeros((len(data_labels), 2, 200, 200))
        labels = torch.zeros(len(data_labels))
        raw_Xs = torch.zeros((len(data_labels), 200, 9))
        mask = []
        for i, name in tqdm(enumerate(data_labels.keys())):
            if name not in data_SC.keys() or name not in data_FC.keys() or name not in data_raw_X.keys():
                continue
            if 'nih_totalcogcomp_ageadjusted' not in data_labels[name].keys():
                continue
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])

            labels[i] = float(data_labels[name]['nih_totalcogcomp_ageadjusted'])

            raw_X = data_raw_X[name].drop(columns=['StructName'])
            raw_X = raw_X.to_numpy().astype(float)
            raw_Xs[i] = torch.tensor(raw_X)

            mask.append(i)
        adjs = adjs[mask]
        labels = labels[mask]
        raw_Xs = raw_Xs[mask]

        mu_lbls, std_lbls = None, None
        if label_type == 'classification':
            labels_class = torch.zeros_like(labels, dtype=torch.long)
            for i, label in enumerate(labels.unique()):
                labels_class[labels == label] = i
            labels = labels_class

        else:
            mu_lbls, std_lbls = labels.mean(), labels.std()
            labels = (labels - mu_lbls) / std_lbls
        
        batchnorm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
        layernorm = nn.LayerNorm([adjs.shape[-2], adjs.shape[-1]], elementwise_affine=False)

        original_feat_shape = raw_Xs.shape
        raw_Xs = batchnorm(
            raw_Xs.reshape(-1, raw_Xs.shape[-1])
        ).reshape(original_feat_shape)

        original_adjs_shape = adjs.shape
        adjs = layernorm(
            adjs.reshape(-1, adjs.shape[-2], adjs.shape[-1])
        ).reshape(original_adjs_shape)

        data = {
            'adjs': adjs,
            'raw_Xs': raw_Xs,
            'labels': labels,
            'mu_lbls': mu_lbls,
            'std_lbls': std_lbls,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        assert (adjs == torch.transpose(adjs, 2, 3)).all().item(), "adj matrices are not symmetric"

    if eval_type == 'split':
        train_size, valid_size, test_size = \
            split_args['train_size'], split_args['valid_size'], split_args['test_size']
        idx = np.arange(len(labels))
        train_valid_idx, test_idx = \
            train_test_split(idx, test_size=test_size)
        train_idx, valid_idx = \
            train_test_split(train_valid_idx, 
                            test_size=valid_size / (train_size + valid_size))
        splits = {
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx,
        }
    elif eval_type == 'cross':
        kfold = KFold(n_splits=5, shuffle=True)
        splits = list(kfold.split(X=idx))

    return adjs, raw_Xs, labels, splits, mu_lbls, std_lbls

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Evaluator:
    def __init__(self, mu_lbls, std_lbls, label_type, num_classes, device):
        assert label_type in ['classification', 'regression']
        self.mu_lbls, self.std_lbls = mu_lbls, std_lbls
        if label_type == 'classification':
            assert num_classes is not None
            self.acc, self.auroc, self.auprc = \
                Accuracy(task="multiclass", num_classes=num_classes).to(device), \
                AUROC(task="multiclass", num_classes=num_classes).to(device), \
                AveragePrecision(task="multiclass", num_classes=num_classes).to(device)
        else:
            self.mse = MeanSquaredError().to(device)
        self.label_type = label_type

    def evaluate(self, logits: torch.Tensor, labels: torch.Tensor):
        if self.label_type == 'classification':
            return self.acc(logits, labels), self.auroc(logits, labels), self.auprc(logits, labels)
        else:
            labels = labels * self.std_lbls + self.mu_lbls
            logits = logits * self.std_lbls + self.mu_lbls
            return self.mse(logits.squeeze(), labels).sqrt()

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def step_score(self, score, model, save=True):  # test score
        if self.best_score is None:
            self.best_score = score
            if save:
                self.save_model(model)
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if save:
                self.save_model(model)
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
            self.early_stop = False

    def save_model(self, model):
        model.eval()
        self.best_model = deepcopy(model.state_dict())

    def load_model(self, model):
        model.load_state_dict(self.best_model)

def evaluate(g, feat, labels, mask, model: torch.nn.Module):
    model.eval()
    with torch.no_grad():
        logits = model(g, feat)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def adj_weight2bin(adjs, ratio_sc, ratio_fc, single_modal=False):
    if single_modal:
        assert ratio_sc == ratio_fc # only one ratio is needed
        topk = int(adjs.shape[-2] * adjs.shape[-1] * ratio_sc)
        original_shape = adjs.shape
        adjs = adjs.flatten(-2)
        idx = torch.topk(adjs, topk, dim=-1)[1]
        adjs = scatter(torch.ones_like(idx), idx).int()
        return adjs.reshape(original_shape)

    if adjs.shape[1] == 2:
        topk_0 = int(adjs.shape[-2] * adjs.shape[-1] * ratio_sc)
        topk_1 = int(adjs.shape[-2] * adjs.shape[-1] * ratio_fc)
        original_shape = [adjs.shape[0], adjs.shape[2], adjs.shape[3]]
        adjs = adjs.flatten(-2)
        idx_0 = torch.topk(adjs[:, 0], topk_0, dim=-1)[1]
        idx_1 = torch.topk(adjs[:, 1], topk_1, dim=-1)[1]
        adjs_0 = scatter(torch.ones_like(idx_0), idx_0).int()
        adjs_1 = scatter(torch.ones_like(idx_1), idx_1).int()
        adjs_0 = adjs_0.reshape(original_shape)
        adjs_1 = adjs_1.reshape(original_shape)

        return adjs_0, adjs_1
    
def pyg_preprocess_sign(raw_Xs: torch.Tensor, adjs: torch.Tensor, k: int):
    data_list = torch.zeros((
        raw_Xs.shape[0], adjs.shape[1], k, raw_Xs.shape[1], raw_Xs.shape[2]
    )).to(raw_Xs.device) # num_nodes, num_graph_layers, k, num_nodes, num_feats

    for i in range(k):
        data_list[:, 0, i, :] = adjs[:, 0, :, :].matmul(raw_Xs)
        data_list[:, 1, i, :] = adjs[:, 1, :, :].matmul(raw_Xs)
        adjs = adjs.matmul(adjs)
    return data_list

def to_pyg_single(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, ratio_sc: float, ratio_fc: float, option: str):
    adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)

    if option == 'sc':
        adjs_target = adjs_0
    elif option == 'fc':
        adjs_target = adjs_1
    
    data_list = []
    for i in tqdm(range(len(adjs_target))):
        data = {
            'x': raw_Xs[i],
            'y': labels[i],
            'edge_index': torch.stack(torch.nonzero(adjs_target[i], as_tuple=True)),
            'adj_sc': adjs_0[i].unsqueeze(0),
            'adj_fc': adjs_1[i].unsqueeze(0),
        }
        data = Data(**data)
        data_list.append(data)

    return data_list

def split_pyg(data_list: list, train_idx: list, valid_idx: list, test_idx: list):
    print("preprocessing pyg data list")
    time_st = time()
    train_data = [data_list[i] for i in train_idx]
    train_data = Batch.from_data_list(train_data).to(device)

    valid_data = [data_list[i] for i in valid_idx]
    valid_data = Batch.from_data_list(valid_data).to(device)

    test_data = [data_list[i] for i in test_idx]
    test_data = Batch.from_data_list(test_data).to(device)

    print(f"finish preprocessing: {time() - time_st:.2f}s")
    return train_data, valid_data, test_data

def to_pyg_fuse(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, 
                fuse_type: str, reduce_fuse: str, 
                ratio_sc: float, ratio_fc: float, ratio: float):
    if fuse_type == 'fuse_graph':
        if reduce_fuse == 'mean':
            adjs = adjs.mean(dim=1)
            adjs = adj_weight2bin(adjs, ratio, ratio, single_modal=True)
        elif reduce_fuse == 'sum':
            adjs = adjs.sum(dim=1)
            adjs = adj_weight2bin(adjs, ratio, ratio, single_modal=True)
        elif reduce_fuse == 'and':
            adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
            adjs = adjs_0 * adjs_1
        elif reduce_fuse == 'or':
            adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
            adjs = torch.logical_or(adjs_0, adjs_1).int()
        data_list = []
        for i in tqdm(range(len(adjs))):
            data = {
                'x': raw_Xs[i],
                'y': labels[i],
                'edge_index': torch.stack(torch.nonzero(adjs[i], as_tuple=True)),
            }
            data = Data(**data)
            data_list.append(data)

    elif fuse_type in ['fuse_embed', 'fuse_pred']:
        adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)
        data_list = []
        for i in tqdm(range(len(adjs_0))):
            data = {
                'x': raw_Xs[i],
                'y': labels[i],
                'edge_index_sc': torch.stack(torch.nonzero(adjs_0[i], as_tuple=True)),
                'edge_index_fc': torch.stack(torch.nonzero(adjs_1[i], as_tuple=True)),
            }
            data = Data(**data)
            data_list.append(data)

    return data_list

def get_fuse_type(model_name: str):
    if 'fuse_embed' in model_name:
        fuse_type = 'fuse_embed'
    elif 'fuse_graph' in model_name:
        fuse_type = 'fuse_graph'
    if 'fuse_pred' in model_name:
        fuse_type = 'fuse_pred'
    return fuse_type

