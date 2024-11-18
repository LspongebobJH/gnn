import torch
import torch.nn as nn
import wandb
import numpy as np

from MHGCN_src import MHGCN
from NeuroPath_src import DetourTransformer, Transformer, GAT, to_pyg, Vanilla
from utils import set_random_seed, load_dataset, model_infer, Evaluator, EarlyStopping

SINGLE_MODALITY_MODELS = ['GCN', 'SAGE', 'SGC', 'GAT', 'Transformer']

def pipe(configs: dict):
    hid_dim = configs['hid_dim']
    nlayers = configs['nlayers']
    lr = configs['lr']
    wd = configs['wd']
    epochs  = configs['epochs']
    patience = configs['patience']
    split_args = configs['split_args']
    use_wandb = configs['use_wandb']
    model_name = configs['model_name']
    reduce = configs['reduce']
    dropout = configs['dropout']
    device='cuda:7'

    label_type = configs.get('label_type', 'regression')
    eval_type = configs.get('eval_type', 'split')

    assert label_type in ['classification', 'regression']
    if label_type == 'classification':
        out_dim = (max(labels)+1).item()
    else:
        out_dim = 1

    adjs, raw_Xs, labels, splits, mu_lbls, std_lbls = \
        load_dataset(split_args=split_args, label_type=label_type, eval_type=eval_type)
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']    
    in_dim = raw_Xs.shape[-1]


    data_list = None
    if model_name == 'MHGCN':
        model = MHGCN(nfeat=in_dim, nlayers=nlayers, nhid=hid_dim, out=out_dim, 
                      adj_shape=[adjs.shape[-2], adjs.shape[-1]])
    elif model_name in ['NeuroPath'] + SINGLE_MODALITY_MODELS:
        ratio_sc = configs.get('ratio_sc', 0.1)
        ratio_fc = configs.get('ratio_fc', 0.5)
        ratio = configs.get('ratio_fc', 0.5)

        if model_name in SINGLE_MODALITY_MODELS:
            data_list = to_pyg(raw_Xs, labels, adjs, ratio_sc=ratio, ratio_fc=ratio, option=configs['modality'])
        else:
            data_list = to_pyg(raw_Xs, labels, adjs, ratio_sc=ratio_sc, ratio_fc=ratio_fc, option='fc')
        
        if model_name == 'NeuroPath':
            model = DetourTransformer(num_nodes = raw_Xs.shape[1], in_dim = in_dim, nclass = out_dim, hiddim = hid_dim, 
                                    nlayer = nlayers)
        elif model_name in ['GCN', 'SAGE', 'SGC']:
            model = Vanilla(model_name=model_name, in_dim=in_dim, hiddim=hid_dim, 
                            nlayers=nlayers, dropout=dropout, reduce=reduce, nclass=out_dim)
        elif model_name == 'GAT':
            model = GAT(in_dim=in_dim, hiddim=hid_dim, nlayers=nlayers, 
                        dropout=dropout, reduce=reduce, nclass=out_dim)
        elif model_name == 'Transformer':
            model = Transformer(in_dim = in_dim, hiddim = hid_dim, nclass = out_dim)
    model = model.cuda()
                                

    if label_type == 'classification':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    earlystop = EarlyStopping(patience)

    evaluator = Evaluator(mu_lbls=mu_lbls, std_lbls=std_lbls, 
                          label_type=label_type, num_classes=out_dim, device=device)

    adjs = adjs.cuda()
    raw_Xs = raw_Xs.cuda()
    labels = labels.cuda()

    best_train_rmse = torch.inf
    best_val_rmse = torch.inf
    best_test_rmse = torch.inf
    cnt = 0
    for epoch in range(epochs):
        model.train()
        if epoch == 1000 and  model_name == 'MHGCN':
            for g in optimizer.param_groups:
                g['lr'] *= 1e-1
        model.train()
        logits = model_infer(model, model_name, adjs=adjs, raw_Xs=raw_Xs, 
                             data_list=data_list, idx=train_idx, device=device)
        loss = loss_fn(logits, labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        if label_type == 'classification':
            train_acc, train_auroc, train_auprc = evaluator.evaluate(logits, labels[train_idx])
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Train Auroc {train_auroc:.4f} | "
                    f"Train Auprc: {train_auprc:.4f}")
            if use_wandb:
                wandb.log({
                    'train_acc': train_acc,
                    'train_auroc': train_auroc,
                    'train_auprc': train_auprc,
                })
        else:
            train_rmse = evaluator.evaluate(logits, labels[train_idx])
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train RMSE {train_rmse:.4f}")
            if use_wandb:
                wandb.log({
                    'train_rmse': train_rmse,
                })

        if epoch % 5 == 0:
            model.eval()
            if label_type == 'classification':
                logits_valid = model_infer(model, model_name, adjs=adjs, raw_Xs=raw_Xs, 
                                           data_list=data_list, idx=valid_idx, device=device)
                logits_test = model_infer(model, model_name, adjs=adjs, raw_Xs=raw_Xs, 
                                          data_list=data_list, idx=test_idx, device=device)
                valid_acc, valid_auroc, valid_auprc = evaluator.evaluate(logits_valid, labels[valid_idx])
                test_acc, test_auroc, test_auprc = evaluator.evaluate(logits_test, labels[test_idx])
                print(f"Valid Acc {valid_acc:.4f} | Valid Auroc {valid_auroc:.4f} | Valid Auprc: {valid_auprc:.4f}")
                print(f"Test Acc {test_acc:.4f} | Test Auroc {test_auroc:.4f} | Test Auprc: {test_auprc:.4f}")
                
                if use_wandb:
                    wandb.log({
                        'valid_acc': valid_acc,
                        'valid_auroc': valid_auroc,
                        'valid_auprc': valid_auprc,

                        'test_acc': test_acc,
                        'test_auroc': test_auroc,
                        'test_auprc': test_auprc,
                    })
            else:
                logits_valid = model_infer(model, model_name, adjs=adjs, raw_Xs=raw_Xs, 
                                           data_list=data_list, idx=valid_idx, device=device)
                logits_test = model_infer(model, model_name, adjs=adjs, raw_Xs=raw_Xs, 
                                         data_list=data_list, idx=test_idx, device=device)
                valid_rmse = evaluator.evaluate(logits_valid, labels[valid_idx])
                test_rmse = evaluator.evaluate(logits_test, labels[test_idx])

                print(f"Valid RMSE {valid_rmse:.4f} | Test RMSE {test_rmse:.4f}")

                if valid_rmse < best_val_rmse:
                    best_train_rmse = train_rmse
                    best_val_rmse = valid_rmse
                    best_test_rmse = test_rmse
                    cnt = 0
                else:
                    cnt += 1

                if use_wandb:
                    wandb.log({
                        'valid_rmse': valid_rmse,
                        'test_rmse': test_rmse,
                    })

                if cnt >= patience:
                    break
    if use_wandb:
        wandb.log({
            'best_train_rmse': best_train_rmse,
            'best_val_rmse': best_val_rmse,
            'best_test_rmse': best_test_rmse,
        })
    return best_train_rmse.item(), best_val_rmse.item(), best_test_rmse.item()
        # earlystop.step_score(val_acc, model)
        # print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
        #       f"Test acc: {test_acc:.4f} | Patience: {earlystop.counter}/{patience}")
        
        # if earlystop.early_stop:
        #     print("Early Stopping!")
        #     break
    # earlystop.load_model(model)
    # acc = evaluate(g, feat, labels, test_mask, model)
    # print("Test accuracy {:.4f}".format(acc))
    # test_acc_list.append(acc)
    # return train_acc_list, valid_acc_list, test_acc_list, acc

if __name__ == '__main__':
    log_idx = 1
    train, valid, test = [], [], []
    for model_name in ['NeuroPath', 'SGC', 'GAT', 'Transformer']:
        model_name = 'GCN'
        for seed in range(5):
            set_random_seed(seed)
            searchSpace = {
                        "hid_dim": 128,
                        "lr": 1e-3,
                        "epochs": 2000,
                        "patience": 10,
                        "wd": 0,
                        "nlayers": 1,
                        "split_args": {
                            'train_size': 0.6,
                            'valid_size': 0.2,
                            'test_size': 0.2,
                        },
                        "dropout": 0.5,
                        "modality": 'sc',
                        "ratio": 0.3,
                        "reduce": "mean",
                        "use_wandb": False,
                        "model_name": model_name,
                    }
            if searchSpace['use_wandb']:
                run = wandb.init(
                    # Set the project where this run will be logged
                    project="multiplex gnn",
                    # Track hyperparameters and run metadata
                    config=searchSpace
                )
            best_train_rmse, best_val_rmse, best_test_rmse = pipe(searchSpace)
            train.append(best_train_rmse)
            valid.append(best_val_rmse)
            test.append(best_test_rmse)

        with open(f'./logs/log_{log_idx}.txt', 'a') as f:
            f.write(f"{searchSpace['model_name']}: ")
            f.write(f'best_train_rmse: {np.mean(train):.4f}±{np.std(train):.4f} | '
                    f'best_val_rmse: {np.mean(valid):.4f}±{np.std(valid):.4f} | '
                    f'best_test_rmse: {np.mean(test):.4f}±{np.std(test):.4f}\n')