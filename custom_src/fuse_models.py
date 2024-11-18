import torch
from torch import nn

from typing import Optional, Tuple, Union

from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv, GINConv, MessagePassing
from torch_scatter import scatter_mean, scatter_max, scatter

import torch.nn.functional as F

class VanillaFuse(nn.Module):
    # siamese network for now
    def __init__(self, model_name, in_dim, hid_dim, nlayers, dropout, 
                 nclass=1, reduce_nodes='mean', reduce_fuse='mean') -> None:
        super().__init__()

        if model_name == 'GCN_fuse':
            model_class = GCNConv
        elif model_name == 'SAGE_fuse':
            model_class = SAGEConv
        elif model_name == 'SGC_fuse':
            model_class = SGConv
        elif model_name == 'GIN_fuse': # not work yet
            model_class = GINConv

        self.dropout = dropout
        self.reduce_nodes = reduce_nodes  # Control reduction method
        self.reduce_fuse = reduce_fuse
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(model_class(in_dim, hid_dim))
        self.net.append(nn.BatchNorm1d(hid_dim))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(model_class(hid_dim, hid_dim))
            self.net.append(nn.BatchNorm1d(hid_dim))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        self.net.append(model_class(hid_dim, hid_dim))

        if self.reduce_fuse == 'concat':
            self.output = nn.Linear(hid_dim * 2, nclass)
        else:
            self.output = nn.Linear(hid_dim, nclass)

    def forward(self, batch):
        x_0, x_1 = batch.x, batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x_0 = net(x_0, batch.edge_index_sc)
                x_1 = net(x_1, batch.edge_index_fc)
            else:
                x_0 = net(x_0)
                x_1 = net(x_1)
        if self.reduce_fuse == 'mean':
            x = (x_0 + x_1) / 2
        elif self.reduce_fuse == 'concat':
            x = torch.concat([x_0, x_1], dim=-1)
        elif self.reduce_fuse == 'sum':
            x = x_0 + x_1
        self.output(x)
        return scatter(x, batch.batch, dim=0, reduce=self.reduce_fuse)

class GAT(nn.Module):
    def __init__(self, in_dim, hid_dim, nlayers, dropout, nclass=1, reduce='mean') -> None:
        super().__init__()
        self.dropout = dropout
        self.reduce = reduce  # Control reduction method
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(GATConv(in_dim, hid_dim, heads=2))
        self.net.append(nn.BatchNorm1d(hid_dim * 2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(GATConv(hid_dim * 2, hid_dim, heads=2))
            self.net.append(nn.BatchNorm1d(hid_dim * 2))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        # Final layer to hidden dimension
        self.net.append(GATConv(hid_dim * 2, nclass, heads=1))

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)

        return scatter(x, batch.batch, dim=0, reduce=self.reduce)