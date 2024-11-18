import torch
from torch_scatter import scatter
from torch_geometric.data import Batch, Data
from utils import adj_weight2bin

from tqdm import tqdm
from time import time

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

def split_pyg(data_list: list, train_idx: list, valid_idx: list, test_idx: list):
    print("preprocessing pyg data list")
    time_st = time()
    train_data = [data_list[i] for i in train_idx]
    train_data = Batch.from_data_list(train_data).cuda()

    valid_data = [data_list[i] for i in valid_idx]
    valid_data = Batch.from_data_list(valid_data).cuda()

    test_data = [data_list[i] for i in test_idx]
    test_data = Batch.from_data_list(test_data).cuda()

    print(f"finish preprocessing: {time() - time_st:.2f}s")
    return train_data, valid_data, test_data

