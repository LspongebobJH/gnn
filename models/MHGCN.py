import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

from .MHGCN_utils import adj_matrix_weight_merge


class GCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(n_layers-2):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))
        self.layers.append(GraphConv(hid_dim, out_dim))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class MHGCN(nn.Module):
    def __init__(self, n_layers, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        assert n_layers >= 2
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_dim, hid_dim, activation=F.relu))
        for _ in range(n_layers-2):
            self.layers.append(GraphConv(hid_dim, hid_dim, activation=F.relu))
        self.layers.append(GraphConv(hid_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

        """
        Set the trainable weight of adjacency matrix aggregation
        """
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(2, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b, a=0, b=0.1)

    def forward(self, feature, A):
        
        
        
        U1 = self.gc1(feature, final_A)
        U2 = self.gc2(U1, final_A)

        # Average pooling
        return (U1+U2)/2