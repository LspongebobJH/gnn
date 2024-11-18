import torch
from torch import nn

from typing import Optional, Tuple, Union

from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv, MessagePassing
from torch_scatter import scatter_mean, scatter_max, scatter

import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, 
        nclass: int = 1,
        nlayer: int = 1,
        num_nodes: int=116,
        in_dim: Union[int, Tuple[int, int]] = 10,
        dropout: float = 0.1,
        hiddim: int = 1024) -> None:
        super().__init__()
        
        heads: int = 2 if in_dim % 2 == 0 else 3
        self.nlayer = nlayer
        self.num_nodes = num_nodes

        self.lin_first = nn.Sequential(
            nn.Linear(in_dim, in_dim), 
            nn.BatchNorm1d(in_dim), 
            nn.LeakyReLU(),
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_dim, hiddim), 
            nn.BatchNorm1d(hiddim), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_dim, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            nlayers=1
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_dim = in_dim
        self.hiddim = hiddim

        self.classifier = Classifier(GCNConv, hiddim, nclass, num_nodes)


    def forward(self, data):
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), data.x.shape[1])
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature)
        h = self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_dim))
        return self.classifier(h, data.edge_index, data.batch).flatten()

class Vanilla(nn.Module):
    def __init__(self, model_name, in_dim, hiddim, nlayers, dropout, nclass=1, reduce='mean') -> None:
        super().__init__()
        
        if model_name == 'GCN':
            model_class = GCNConv
        if model_name == 'SAGE':
            model_class = SAGEConv
        if model_name == 'SGC':
            model_class = SGConv

        self.dropout = dropout
        self.reduce = reduce  # Control reduction method
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(model_class(in_dim, hiddim))
        self.net.append(nn.BatchNorm1d(hiddim))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(model_class(hiddim, hiddim))
            self.net.append(nn.BatchNorm1d(hiddim))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        # Final layer to hidden dimension
        self.net.append(model_class(hiddim, nclass))

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return scatter(x, batch.batch, dim=0, reduce=self.reduce)

class GAT(nn.Module):
    def __init__(self, in_dim, hiddim, nlayers, dropout, nclass=1, reduce='mean') -> None:
        super().__init__()
        self.dropout = dropout
        self.reduce = reduce  # Control reduction method
        self.net = nn.ModuleList()

        # Input layer
        self.net.append(GATConv(in_dim, hiddim, heads=2))
        self.net.append(nn.BatchNorm1d(hiddim * 2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Dropout(p=dropout))

        # Hidden layers
        for _ in range(nlayers - 1):
            self.net.append(GATConv(hiddim * 2, hiddim, heads=2))
            self.net.append(nn.BatchNorm1d(hiddim * 2))
            self.net.append(nn.ReLU())
            self.net.append(nn.Dropout(p=dropout))

        # Final layer to hidden dimension
        self.net.append(GATConv(hiddim * 2, nclass, heads=1))

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            x = net(x)

        return scatter(x, batch.batch, dim=0, reduce=self.reduce)




class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, num_nodes, aggr='learn', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.ModuleList([
            net(feat_dim, feat_dim),
            nn.LeakyReLU(),
            net(feat_dim, nclass)
        ])
        if isinstance(self.net[0], MessagePassing):
            self.nettype = 'gnn'
        else:
            self.nettype = 'mlp'
        self.aggr = aggr
        if aggr == 'learn':
            self.pool = nn.Sequential(nn.Linear(num_nodes, 1), nn.LeakyReLU())
        elif aggr == 'mean':
            self.pool = scatter_mean
        elif aggr == 'max':
            self.pool = scatter_max
        
    
    def forward(self, x, edge_index, batch):
        if self.nettype == 'gnn':
            x = self.net[0](x, edge_index)
            x = self.net[1](x)
            x = self.net[2](x, edge_index)
        else:
            x = self.net[0](x)
            x = self.net[1](x)
            x = self.net[2](x)
    
        if self.aggr == 'learn':
            x = self.pool(x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2))[..., 0]
        else:
            if self.aggr == 'max': 
                x = x.view(batch.max()+1, len(torch.where(batch==0)[0]), x.shape[-1]).transpose(-1, -2).max(-1)[0]
            else:
                x = self.pool(x, batch, dim=0)
        return x