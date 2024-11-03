import os
import pickle
import numpy as np
import torch
from copy import deepcopy
from tqdm import tqdm
import random

from torchmetrics import Accuracy, AUROC, AveragePrecision, MeanSquaredError
from sklearn.model_selection import train_test_split

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
    elif eval_type == 'cross':
        assert cross_args is not None

    data_path = '/home/jiahang/gnn/dataset'
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
    ## we only take graph which have labels.
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

    if label_type == 'classification':
        labels_class = torch.zeros_like(labels, dtype=torch.long)
        for i, label in enumerate(labels.unique()):
            labels_class[labels == label] = i
        labels = labels_class

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

    return adjs, raw_Xs, labels, splits

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class Evaluator:
    def __init__(self, label_type, num_classes, device):
        assert label_type in ['classification', 'regression']
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