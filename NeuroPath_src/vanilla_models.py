import torch
from torch import nn

from typing import Optional, Tuple, Union

from torch_geometric.nn import GCNConv, SAGEConv, SGConv, GATConv, MessagePassing
from torch_scatter import scatter_mean, scatter_max

import torch.nn.functional as F

class Transformer(nn.Module):

    def __init__(self, 
        nclass: int = 1,
        nlayer: int = 1,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        dropout: float = 0.1,
        hiddim: int = 1024) -> None:
        super().__init__()
        
        heads: int = 2 if in_channel % 2 == 0 else 3
        self.nlayer = nlayer
        self.node_sz = node_sz

        self.lin_first = nn.Sequential(
            nn.Linear(in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU(),
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, hiddim), 
            nn.BatchNorm1d(hiddim), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.hiddim = hiddim

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, data):
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), data.x.shape[1])
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature)
        h = self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))
        return self.classifier(h, data.edge_index, data.batch).flatten()


class GCN(nn.Module):
    def __init__(self, in_channel, hiddim, node_sz, nclass=1) -> None:
        super().__init__()
        self.dropout = 0.3
        self.net = nn.ModuleList([
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, in_channel),
            nn.LeakyReLU(),
            GCNConv(in_channel, hiddim),
            nn.LeakyReLU(),
        ])

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = F.dropout(x, self.dropout, training=self.training)
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return self.classifier(x, batch.edge_index, batch.batch).flatten()


class SAGE(nn.Module):
    def __init__(self, in_channel, hiddim, node_sz, nclass=1) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SAGEConv(in_channel, hiddim),
            nn.BatchNorm1d(hiddim),
            nn.LeakyReLU(),
        ])

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return self.classifier(x, batch.edge_index, batch.batch).flatten()


class SGC(nn.Module):
    def __init__(self, in_channel, hiddim, node_sz, nclass=1) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, in_channel),
            nn.BatchNorm1d(in_channel),
            nn.LeakyReLU(),
            SGConv(in_channel, hiddim),
            nn.BatchNorm1d(hiddim),
            nn.LeakyReLU(),
        ])

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return self.classifier(x, batch.edge_index, batch.batch).flatten()

class GAT(nn.Module):
    def __init__(self, in_channel, hiddim, node_sz, nclass=1) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            GATConv(in_channel, in_channel, heads=2),
            nn.BatchNorm1d(in_channel * 2),
            nn.LeakyReLU(),
            GATConv(in_channel * 2, in_channel, heads=2),
            nn.BatchNorm1d(in_channel * 2),
            nn.LeakyReLU(),
            GATConv(in_channel * 2, in_channel, heads=2),
            nn.BatchNorm1d(in_channel * 2),
            nn.LeakyReLU(),
            GATConv(in_channel * 2, hiddim, heads=1),
            nn.BatchNorm1d(hiddim),
            nn.LeakyReLU(),
        ])

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, batch):
        x = batch.x
        for net in self.net:
            if isinstance(net, MessagePassing):
                x = net(x, batch.edge_index)
            else:
                x = net(x)
        return self.classifier(x, batch.edge_index, batch.batch).flatten()




class Classifier(nn.Module):

    def __init__(self, net: callable, feat_dim, nclass, node_sz, aggr='learn', *args, **kwargs) -> None:
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
            self.pool = nn.Sequential(nn.Linear(node_sz, 1), nn.LeakyReLU())
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