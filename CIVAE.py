import datetime, os, torch
import numpy as np
from sklearn.model_selection import train_test_split

import CIVAE_src.ci_ivae_main as CI_iVAE

import torch
import torch.nn as nn
import wandb

from MHGCN_src import MHGCN
from utils import set_random_seed, load_dataset, Evaluator, EarlyStopping
from sklearn import svm

def pipe(configs):
    x_concat_adj = configs['x_concat_adj']
    epochs  = configs['epochs']
    split_args = configs['split_args']
    device='cuda:0'
    label_type = 'regression'

    assert label_type in ['classification', 'regression']
    if label_type == 'classification':
        out_dim = (max(labels)+1).item()
    else:
        out_dim = 1

    adjs, raw_Xs, labels, splits = load_dataset(split_args=split_args, label_type=label_type)
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']    

    # raw_Xs = raw_Xs.reshape(raw_Xs.shape[0] * raw_Xs.shape[1], -1)
    raw_Xs = raw_Xs.reshape(-1, raw_Xs.shape[1] * raw_Xs.shape[2])
    batch_norm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
    raw_Xs = batch_norm(raw_Xs)
    
    labels = torch.unsqueeze(labels, 1)

    layer_norm = nn.LayerNorm(normalized_shape=(adjs.shape[-2], adjs.shape[-1]), elementwise_affine=False)
    adjs = layer_norm(adjs)

    if x_concat_adj:
        raw_Xs = torch.concatenate([raw_Xs, adjs.flatten(1)], dim=-1)
    in_dim = raw_Xs.shape[-1]
    

    evaluator = Evaluator(label_type=label_type, num_classes=out_dim, device=device)
    
    # build CI-iVAE networks
    dim_x, dim_u = in_dim, 1
    ci_ivae = CI_iVAE.model(dim_x=dim_x, dim_u=dim_u)

    # train CI-iVAE networks. Results will be saved at the result_path
    now = datetime.datetime.now()
    result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
    CI_iVAE.fit(model=ci_ivae, x_train=raw_Xs[train_idx], u_train=labels[train_idx],
                x_val=raw_Xs[valid_idx], u_val=labels[valid_idx], 
                num_epoch=epochs, result_path=result_path, num_worker=2,
                )

    # extract features with trained CI-iVAE networks
    z_train = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[train_idx])
    z_valid = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[valid_idx])
    z_test = CI_iVAE.extract_feature(result_path=result_path, x=raw_Xs[test_idx])
    z_train = z_train.detach().cpu().numpy()
    z_valid = z_valid.detach().cpu().numpy()
    z_test = z_test.detach().cpu().numpy()

    # evaluation with svr
    svr = svm.SVR()
    svr.fit(z_train, labels[train_idx].flatten().numpy())

    pred_train = svr.predict(z_train)
    pred_valid = svr.predict(z_valid)
    pred_test = svr.predict(z_test)

    pred_train, pred_valid, pred_test = \
        torch.tensor(pred_train), \
        torch.tensor(pred_valid), \
        torch.tensor(pred_test)
    evaluator = Evaluator(label_type=label_type, num_classes=out_dim, device='cpu')
    train_rmse = evaluator.evaluate(pred_train, labels[train_idx].flatten())
    valid_rmse = evaluator.evaluate(pred_valid, labels[valid_idx].flatten())
    test_rmse = evaluator.evaluate(pred_test, labels[test_idx].flatten())

    print(f"train rmse {train_rmse:.4f} | valid rmse {valid_rmse:.4f} | test rmse {test_rmse:.4f}")


if __name__ == '__main__':
    set_random_seed(0)
    searchSpace = {
                "hid_dim": 64,
                "l": 3,
                "lr": 1e-2,
                "epochs": 1,
                "patience": 20,
                "wd": 1e-2,
                "split_args": {
                    'train_size': 0.6,
                    'valid_size': 0.2,
                    'test_size': 0.2,
                },
                "use_wandb": True,
                "x_concat_adj": True
            }
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="multiplex gnn",
    #     # Track hyperparameters and run metadata
    #     config=searchSpace
    # )
    pipe(searchSpace)

# from main repo tutorial
# if __name__ == '__main__':

#     n_train, n_test = 4000, 1000
#     dim_x, dim_u = 100, 5

#     x_train = torch.tensor(np.random.uniform(0.0, 1.0, (n_train, dim_x)), dtype=torch.float32)
#     u_train = torch.tensor(np.random.uniform(0.0, 1.0, (n_train, dim_u)), dtype=torch.float32)
#     x_test = torch.tensor(np.random.uniform(0.0, 1.0, (n_test, dim_x)), dtype=torch.float32)
#     u_test = torch.tensor(np.random.uniform(0.0, 1.0, (n_test, dim_u)), dtype=torch.float32)

#     x_train, x_val, u_train, u_val = train_test_split(x_train, u_train, test_size=(1/6))

#     # make result folder
#     now = datetime.datetime.now()
#     result_path = './results/ci_ivae-time=%d-%d-%d-%d-%d' % (now.month, now.day, now.hour, now.minute, now.second)
#     os.makedirs(result_path, exist_ok=True)
#     print('result_path: ', result_path)

#     # build CI-iVAE networks
#     dim_x, dim_u = np.shape(x_train)[1], np.shape(u_train)[1]
#     ci_ivae = CI_iVAE.model(dim_x=dim_x, dim_u=dim_u)

#     # train CI-iVAE networks. Results will be saved at the result_path
#     CI_iVAE.fit(model=ci_ivae, x_train=x_train, u_train=u_train,
#                 x_val=x_val, u_val=u_val, num_epoch=5, result_path=result_path, num_worker=2)

#     # extract features with trained CI-iVAE networks
#     z_train = CI_iVAE.extract_feature(result_path=result_path, x=x_train)
#     z_test = CI_iVAE.extract_feature(result_path=result_path, x=x_test)
#     z_train = z_train.detach().cpu().numpy()
#     z_test = z_test.detach().cpu().numpy()