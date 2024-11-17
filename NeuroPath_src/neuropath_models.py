import torch
from torch import nn

from typing import Optional, Tuple, Union
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing, GCNConv


from .vanilla_models import Classifier


class DetourTransformer(nn.Module):

    def __init__(self, 
                nclass,
                heads: int = 2,
                nlayer: int = 1,
                node_sz: int=116,
                in_channel: Union[int, Tuple[int, int]] = 10,
                dropout: float = 0.1,
                hiddim: int = 1024,
                detour_type = 'node',
                ) -> None:
        
        super(DetourTransformer, self).__init__()
        org_in_channel = in_channel
        if in_channel % heads != 0:
            in_channel = in_channel  + heads - (in_channel % heads)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz

        # self.batchnorm = nn.BatchNorm1d(org_in_channel)
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, hiddim), 
            nn.BatchNorm1d(hiddim), 
            nn.LeakyReLU(),
        )
        
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None
        ) for _ in range(nlayer)])
        self.net_fc = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.fcsc_loss = nn.MSELoss()

        self.classifier = Classifier(GCNConv, hiddim, nclass, node_sz)

    def forward(self, data):
        node_feature = data.x
        # node_feature = self.batchnorm(node_feature)
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_channel)

        adj = data.adj_sc
        adj_fc = data.adj_fc
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            # mask = torch.zeros(len(adj), adj.shape[1], adj.shape[2], device=adj.device) - torch.inf
            # mask[torch.logical_and(adj, adj_fc)] = 0
            
            # TODO(jiahang): numeric error, weird. Codes are different from paper's descriptions
            mask = torch.zeros(len(adj), adj.shape[1], adj.shape[2], device=adj.device)
            mask[torch.logical_and(adj, adj_fc)] = 1
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)
        
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature, mask=multi_mask)
        
        # return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))
        h = self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))
        return self.classifier(h, data.edge_index, data.batch).flatten()


class DetourTransformerSingleFC(nn.Module):

    def __init__(self, 
        heads: int = 2,
        nlayer: int = 1,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        hiddim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        
        super(DetourTransformerSingleFC, self).__init__()
        org_in_channel = in_channel
        if in_channel % heads != 0:
            in_channel = in_channel  + heads - (in_channel % heads)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )
        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])

        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, node_sz, node_sz) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)

    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_channel)

        adj_fc = data.adj_fc
        adj_fc[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        mask_fc = self.mask_heldout[:len(adj_fc)]
        mask_fc[adj_fc] = 0
        mask_fc = mask_fc.repeat(self.heads, 1, 1)
        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature, mask=mask_fc)

        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))



class DetourTransformerSingleSC(nn.Module):

    def __init__(self, 
        heads: int = 2,
        nlayer: int = 1,
        node_sz: int=116,
        in_channel: Union[int, Tuple[int, int]] = 10,
        out_channel: int = 10,
        concat: bool = False,
        dek: int = 4,
        pek: int = 10,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        hiddim: int = 1024,
        detour_type = 'node',
        batch_size = 32,
        device='cuda:0',
        *args, **kwargs) -> None:
        
        super(DetourTransformerSingleSC, self).__init__()
        org_in_channel = in_channel
        if in_channel % heads != 0:
            in_channel = in_channel  + heads - (in_channel % heads)
        self.detour_type = detour_type
        self.nlayer = nlayer
        self.node_sz = node_sz
            
        self.lin_first = nn.Sequential(
            nn.Linear(org_in_channel, in_channel), 
            nn.BatchNorm1d(in_channel), 
            nn.LeakyReLU()
        )
        self.lin_in = nn.Sequential(
            nn.Linear(in_channel, out_channel), 
            nn.BatchNorm1d(out_channel), 
            nn.LeakyReLU(),
        )

        self.net = nn.ModuleList([torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=in_channel, nhead=heads, dim_feedforward=hiddim, dropout=dropout, batch_first=True),
            num_layers=1,
            norm=None#nn.LayerNorm(in_channel)
        ) for _ in range(nlayer)])
        self.heads = heads
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mask_heldout = torch.zeros(batch_size, node_sz, node_sz) - torch.inf
        self.mask_heldout = self.mask_heldout.to(device)


    def forward(self, data):
        self.loss = 0
        node_feature = data.x
        node_feature = self.lin_first(node_feature)
        node_feature = node_feature.view(data.batch.max()+1, len(torch.where(data.batch==0)[0]), self.in_channel)

        adj = data.adj_sc
        adj_fc = data.adj_fc
        adj[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        adj_fc[:, torch.arange(self.node_sz), torch.arange(self.node_sz)] = True
        org_adj = adj
        multi_mask = []
        for _ in range(self.heads):
            mask = self.mask_heldout[:len(adj)]
            mask[torch.logical_and(adj, adj_fc)] = 0
            adj = (adj.float() @ org_adj.float()) > 0
            multi_mask.append(mask)
        multi_mask = torch.cat(multi_mask)

        for i in range(self.nlayer):
            node_feature = self.net[i](node_feature, mask=multi_mask)

        return self.lin_in(node_feature.reshape(node_feature.shape[0] * node_feature.shape[1], self.in_channel))

    