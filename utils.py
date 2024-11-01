import os
import pickle
import numpy as np
import torch
from copy import deepcopy

def load_dataset():
    data_path = '/Users/jiahang/Documents/gnn/dataset'
    path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
    with open(path, 'rb') as f:
        data_FC = pickle.load(f)

    path = os.path.join(data_path, 'SC.pkl')
    with open(path, 'rb') as f:
        data_SC = pickle.load(f)

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