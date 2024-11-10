import torch
from torch_scatter import scatter
from torch_geometric.data import Batch, Data

from tqdm import tqdm

def to_pyg(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, ratio_sc: float, ratio_fc: float, option: str):
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