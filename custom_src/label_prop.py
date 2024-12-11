import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
import numpy as np

from dgl import DGLGraph

class LabelProp(nn.Module):
    def __init__(self, fuse_method="mean"):
        super().__init__()
        self.fuse_method = fuse_method
        self.simple_prim = ['mean', 'sum', 'min', 'max']
    
    def fusion(self, lbls: torch.Tensor, knn_g: DGLGraph, null_mask: torch.Tensor):
        lbls = lbls.clone()
        if null_mask.sum() > 0:
            if self.fuse_method in self.simple_prim:
                knn_n, target_n = knn_g.in_edges(v=torch.where(null_mask)[0])
                lbls[null_mask] = scatter(lbls[knn_n], target_n, 
                                        reduce=self.fuse_method, dim=0, 
                                        dim_size=lbls.shape[0])[null_mask]

        return lbls

    def forward(self, lbls, no_lbl_idx, knn_sc, knn_fc):
        lbls1 = self.fusion(lbls, knn_sc, no_lbl_idx)
        lbls2 = self.fusion(lbls, knn_fc, no_lbl_idx)

        return (lbls1 + lbls2) / 2.

