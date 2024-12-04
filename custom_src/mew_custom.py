import torch
from torch import nn

from torch_scatter import scatter
import numpy as np

from .mew import SIGN_pred, SIGN_v2

from dgl import DGLError, remove_self_loop, remove_edges, EID
from dgl.base import dgl_warning
from dgl.transforms.functional import pairwise_squared_distance
from dgl.sampling import sample_neighbors
from dgl.transforms.functional import convert
import dgl.backend as dglF


class SIGNv2Custom(SIGN_v2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, adjs: torch.Tensor, feats: torch.Tensor):
        """
        Different from original sign v2:
        original sign v2: given [X, AX, A2X, A3X], apply NN to each, concate, NN transform then output
        current sign v2: given X, apply adj @ X @ NN layer by layer, concate, NN transform then output
        in other words, current sign v2 is online version of original sign v2
        """
        hidden = []
        for i, ff in enumerate(self.inception_ffs):
            feats = adjs.matmul(feats)
            h = feats.reshape(feats.shape[0] * feats.shape[1], -1)
            h = ff(h)
            hidden.append(self.batch_norms[i](h))
        out = self.project(self.dropout(self.prelu(torch.cat(hidden, dim=-1))))
        return out
        
class MewCustom(SIGN_pred):
    def __init__(self, num_layer, num_feat, emb_dim, drop_ratio, shared, 
                 k=None, supp_pool="mean", fuse_type='graph_embed',
                 *args, **kwargs):
        super().__init__(num_layer, num_feat, emb_dim, drop_ratio=drop_ratio, shared=shared, *args, **kwargs)
        self.k = k
        self.fuse_type = fuse_type
        self.sign = SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)
        self.sign2 = self.sign if shared else SIGNv2Custom(num_layer, num_feat, emb_dim, dropout=drop_ratio)

        self.supp_pool = supp_pool

    def fuse_graph_embed(self, node_embed_1, node_embed_2, batch, no_sc_idx, no_fc_idx):
        graph_embed_1 = self.pool(node_embed_1, batch.to(node_embed_1.device))
        graph_embed_2 = self.pool(node_embed_2, batch.to(node_embed_2.device)) 

        if no_sc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_2, no_sc_idx, self.k, exclude_self=True) # must be graph_embed_2 not embed_1
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_sc_idx)[0])
            graph_embed_1[no_sc_idx] = scatter(graph_embed_1[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=graph_embed_1.shape[0])[no_sc_idx]
        if no_fc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_1, no_fc_idx, self.k, exclude_self=True)  # must be graph_embed_1 not embed_2
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_fc_idx)[0])
            graph_embed_2[no_fc_idx] = scatter(graph_embed_2[knn_n], target_n, 
                                                reduce=self.supp_pool, dim=0, 
                                                dim_size=graph_embed_2.shape[0])[no_fc_idx]
            
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(graph_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(graph_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(graph_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(graph_embed_2, self.attention))

        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        graph_embed = (values[:,0].unsqueeze(1) * graph_embed_1) + (values[:,1].unsqueeze(1) * graph_embed_2)

        return graph_embed
    
    def fuse_node_embed_on_graph_embed(self, node_embed_1, node_embed_2, 
                                          batch, no_sc_idx, no_fc_idx, 
                                          num_graphs, num_nodes):

        graph_embed_1 = self.pool(node_embed_1, batch.to(node_embed_1.device))
        graph_embed_2 = self.pool(node_embed_2, batch.to(node_embed_2.device))

        node_embed_1 = node_embed_1.reshape(num_graphs, num_nodes, -1)
        node_embed_2 = node_embed_2.reshape(num_graphs, num_nodes, -1)

        if no_sc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_2, no_sc_idx, self.k, exclude_self=True) # must be graph_embed_2 not embed_1
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_sc_idx)[0])
            node_embed_1[no_sc_idx] = scatter(node_embed_1[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=node_embed_1.shape[0])[no_sc_idx]
        if no_fc_idx.sum() > 0:
            knn_g = knn_graph(graph_embed_1, no_fc_idx, self.k, exclude_self=True)  # must be graph_embed_1 not embed_2
            knn_n, target_n = knn_g.in_edges(v=torch.where(no_fc_idx)[0])
            node_embed_2[no_fc_idx] = scatter(node_embed_2[knn_n], target_n, 
                                            reduce=self.supp_pool, dim=0, 
                                            dim_size=node_embed_2.shape[0])[no_fc_idx]
            
        node_embed_1 = node_embed_1.reshape(num_graphs * num_nodes, -1)
        node_embed_2 = node_embed_2.reshape(num_graphs * num_nodes, -1)
            
        if self.attn_weight:
            geom_ = self.leakyrelu(torch.mm(self.w1(node_embed_1), self.attention))
            cell_type_ = self.leakyrelu(torch.mm(self.w2(node_embed_2), self.attention))
        else:
            geom_ = self.leakyrelu(torch.mm(node_embed_1, self.attention))
            cell_type_ = self.leakyrelu(torch.mm(node_embed_2, self.attention))
        
        values = torch.softmax(torch.cat((geom_, cell_type_), dim=1), dim=1)
        node_embed = (values[:,0].unsqueeze(1) * node_embed_1) + (values[:,1].unsqueeze(1) * node_embed_2)
        graph_embed = self.pool(node_embed, batch.to(node_embed_1.device))

        return graph_embed

    def forward(self, adjs, feats, no_sc_idx, no_fc_idx):
        batch = torch.repeat_interleave(
            torch.arange(feats.shape[0]), 
            feats.shape[-2]
        ).to(feats.device)
        num_graphs, num_nodes = feats.shape[0], feats.shape[1]
        node_embed_1 = self.sign(adjs[:, 0], feats) # geom
        node_embed_2 = self.sign2(adjs[:, 1], feats) # cell_type

        if self.fuse_type == 'graph_embed':
            graph_embed = \
                self.fuse_graph_embed(node_embed_1, node_embed_2, batch, no_sc_idx, no_fc_idx)
       
        elif self.fuse_type == 'node_embed_on_graph_embed':
            graph_embed = \
                self.fuse_node_embed_on_graph_embed(node_embed_1, node_embed_2, 
                                                    batch, no_sc_idx, no_fc_idx, 
                                                    num_graphs, num_nodes)
        if self.num_graph_tasks > 0:
            graph_pred = self.graph_pred_module(graph_embed)

        return graph_pred
    
"""
adapted from dgl.knn_graph, but add some constraints
"""

def knn_graph(
    x, null_idx, k, algorithm="bruteforce-blas", dist="euclidean", exclude_self=False
):
    if exclude_self:
        # add 1 to k, for the self edge, since it will be removed
        k = k + 1

    # check invalid k
    if k <= 0:
        raise DGLError("Invalid k value. expect k > 0, got k = {}".format(k))

    # check empty point set
    x_size = tuple(dglF.shape(x))
    if x_size[0] == 0:
        raise DGLError("Find empty point set")

    d = dglF.ndim(x)
    x_seg = x_size[0] * [x_size[1]] if d == 3 else [x_size[0]]
    if algorithm == "bruteforce-blas":
        result = _knn_graph_blas(x, null_idx, k, dist=dist)

    if exclude_self:
        # remove_self_loop will update batch_num_edges as needed
        result = remove_self_loop(result)

        # If there were more than k(+1) coincident points, there may not have been self loops on
        # all nodes, in which case there would still be one too many out edges on some nodes.
        # However, if every node had a self edge, the common case, every node would still have the
        # same degree as each other, so we can check that condition easily.
        # The -1 is for the self edge removal.
        clamped_k = min(k, np.min(x_seg)) - 1
        if result.num_edges() != clamped_k * result.num_nodes():
            # edges on any nodes with too high degree should all be length zero,
            # so pick an arbitrary one to remove from each such node
            degrees = result.in_degrees()
            node_indices = dglF.nonzero_1d(degrees > clamped_k)
            edges_to_remove_graph = sample_neighbors(
                result, node_indices, 1, edge_dir="in"
            )
            edge_ids = edges_to_remove_graph.edata[EID]
            result = remove_edges(result, edge_ids)

    return result



def _knn_graph_blas(x, null_idx, k, dist="euclidean"):
    if dglF.ndim(x) == 2:
        x = dglF.unsqueeze(x, 0)
    n_samples, n_points, _ = dglF.shape(x)

    if k > n_points:
        dgl_warning(
            "'k' should be less than or equal to the number of points in 'x'"
            "expect k <= {0}, got k = {1}, use k = {0}".format(n_points, k)
        )
        k = n_points

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == "cosine":
        l2_norm = lambda v: dglF.sqrt(dglF.sum(v * v, dim=2, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)

    ctx = dglF.context(x)
    dist = pairwise_squared_distance(x)
    # Jiahang: revise such that null graphs not in neighbors
    null_idx_2d = (null_idx.unsqueeze(-1).float() @ null_idx.unsqueeze(0).float()).bool()
    dist[:, null_idx_2d] = torch.inf
    dist[:, null_idx, null_idx] = 0.

    k_indices = dglF.astype(dglF.argtopk(dist, k, 2, descending=False), dglF.int64)
    # index offset for each sample
    offset = dglF.arange(0, n_samples, ctx=ctx) * n_points
    offset = dglF.unsqueeze(offset, 1)
    src = dglF.reshape(k_indices, (n_samples, n_points * k))
    src = dglF.unsqueeze(src, 0) + offset
    dst = dglF.repeat(dglF.arange(0, n_points, ctx=ctx), k, dim=0)
    dst = dglF.unsqueeze(dst, 0) + offset
    return convert.graph((dglF.reshape(src, (-1,)), dglF.reshape(dst, (-1,))))
            
            
        
        


