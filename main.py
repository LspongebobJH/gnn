import torch
import torch.nn as nn
import wandb

from models import MHGCN
from utils import set_random_seed, load_dataset, Evaluator, EarlyStopping

def pipe(gName, model_name, hid_dim, l, lr, wd, epochs, patience, dropout, split_args, mean_method=None, device='cuda:7'):
    hid_dim = 32
    nlayers = 2
    label_type = 'regression'
    assert label_type in ['classification', 'regression']
    if label_type == 'classification':
        out_dim = (max(labels)+1).item()
    else:
        out_dim = 1

    # split_args = {
    #     'train_size': 0.6,
    #     'valid_size': 0.2,
    #     'test_size': 0.2,
    # }
    adjs, raw_Xs, labels, splits = load_dataset(split_args=split_args, label_type=label_type)
    train_idx, valid_idx, test_idx = splits['train_idx'], splits['valid_idx'], splits['test_idx']    
    in_dim = raw_Xs.shape[-1]

    model = MHGCN(nfeat=in_dim, nlayers=nlayers, nhid=hid_dim, out=out_dim, adj_shape=[adjs.shape[-2], adjs.shape[-1]]).to(device)
    if label_type == 'classification':
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    earlystop = EarlyStopping(patience)

    evaluator = Evaluator(label_type=label_type, num_classes=out_dim, device=device)

    adjs = adjs.to(device)
    raw_Xs = raw_Xs.to(device)
    labels = labels.to(device)

    for epoch in range(epochs):
        if epoch == 1000:
            for g in optimizer.param_groups:
                g['lr'] = 1e-3
        model.train()
        logits = model(adjs[train_idx], raw_Xs[train_idx])
        loss = loss_fn(logits, labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if label_type == 'classification':
            train_acc, train_auroc, train_auprc = evaluator.evaluate(logits, labels[train_idx])
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Train Auroc {train_auroc:.4f} | "
                    f"Train Auprc: {train_auprc:.4f}")
            wandb.log({
                'train_acc': train_acc,
                'train_auroc': train_auroc,
                'train_auprc': train_auprc,
            })
        else:
            train_rmse = evaluator.evaluate(logits, labels[train_idx])
            print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train RMSE {train_rmse:.4f}")
            wandb.log({
                'train_rmse': train_rmse,
            })

        if epoch % 5 == 0:
            model.eval()
            if label_type == 'classification':
                logits_valid = model(adjs[valid_idx], raw_Xs[valid_idx])
                logits_test = model(adjs[test_idx], raw_Xs[test_idx])
                valid_acc, valid_auroc, valid_auprc = evaluator.evaluate(logits_valid, labels[valid_idx])
                test_acc, test_auroc, test_auprc = evaluator.evaluate(logits_test, labels[test_idx])
                print(f"Valid Acc {valid_acc:.4f} | Valid Auroc {valid_auroc:.4f} | Valid Auprc: {valid_auprc:.4f}")
                print(f"Test Acc {test_acc:.4f} | Test Auroc {test_auroc:.4f} | Test Auprc: {test_auprc:.4f}")

                wandb.log({
                    'valid_acc': valid_acc,
                    'valid_auroc': valid_auroc,
                    'valid_auprc': valid_auprc,

                    'test_acc': test_acc,
                    'test_auroc': test_auroc,
                    'test_auprc': test_auprc,
                })
            else:
                logits_valid = model(adjs[valid_idx], raw_Xs[valid_idx])
                logits_test = model(adjs[test_idx], raw_Xs[test_idx])
                valid_rmse = evaluator.evaluate(logits_valid, labels[valid_idx])
                test_rmse = evaluator.evaluate(logits_test, labels[test_idx])
                print(f"Valid RMSE {valid_rmse:.4f} | Test RMSE {test_rmse:.4f}")

                wandb.log({
                    'valid_rmse': valid_rmse,
                    'test_rmse': test_rmse,
                })
        
        # earlystop.step_score(val_acc, model)
        # print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
        #       f"Test acc: {test_acc:.4f} | Patience: {earlystop.counter}/{patience}")
        
        if earlystop.early_stop:
            print("Early Stopping!")
            break
    earlystop.load_model(model)
    # acc = evaluate(g, feat, labels, test_mask, model)
    # print("Test accuracy {:.4f}".format(acc))
    # test_acc_list.append(acc)
    # return train_acc_list, valid_acc_list, test_acc_list, acc

if __name__ == '__main__':
    set_random_seed(0)
    searchSpace = {
                "gName": 'cora',
                "model_name": "gcn",
                "hid_dim": 64,
                "l": 3,
                "lr": 1e-2,
                "epochs": 3000,
                "patience": 20,
                "dropout": 0.0,
                "wd": 1e-2,
                "mean_method": "normal",
                "split_args": {
                    'train_size': 0.6,
                    'valid_size': 0.2,
                    'test_size': 0.2,
                }
            }
    run = wandb.init(
        # Set the project where this run will be logged
        project="multiplex gnn",
        # Track hyperparameters and run metadata
        config=searchSpace
    )
    pipe(**searchSpace)