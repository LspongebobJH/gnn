import torch
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from utils import adj_weight2bin

from tqdm import tqdm

def to_pyg(raw_Xs: torch.Tensor, labels: torch.Tensor, adjs: torch.Tensor, ratio_sc: float, ratio_fc: float, option: str):
    adjs_0, adjs_1 = adj_weight2bin(adjs, ratio_sc, ratio_fc)

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