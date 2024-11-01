import torch
import torch.nn as nn

from utils import load_dataset, EarlyStopping
from models import MHGCN

def pipe(gName, model_name, hid_dim, l, lr, wd, epochs, patience, dropout, mean_method=None, device='cuda', init='virgo'):
    g, data = load_dataset()
    if gName in ['arxiv', 'proteins']:
        feat = g.ndata['feat']
        _, labels = data[0]
        masks = data.get_idx_split()
        train_mask, val_mask, test_mask = masks['train'], masks['valid'], masks['test']
        labels, train_mask, val_mask, test_mask = \
            labels.squeeze().to(device), train_mask.to(device), val_mask.to(device), test_mask.to(device)
    else:   
        feat = g.ndata['feat']
        labels = g.ndata['label']
        masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']
        train_mask = masks[0]
        val_mask = masks[1]
        test_mask = masks[2]
        
    # create GCN model    
    in_dim = feat.shape[1]
    out_dim = data.num_classes
    model = MHGCN(nfeat=in_dim, nhid=hid_dim, out=out_dim).to(device)


    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    earlystop = EarlyStopping(patience)

    # training loop
    train_acc_list, valid_acc_list, test_acc_list = [], [], []
    for epoch in range(epochs):
        model.train()
        logits = model(g, feat)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc = evaluate(g, feat, labels, train_mask, model)
        val_acc = evaluate(g, feat, labels, val_mask, model)
        test_acc = evaluate(g, feat, labels, test_mask, model)
        train_acc_list.append(train_acc)
        valid_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        earlystop.step_score(val_acc, model)
        print(f"Epoch {epoch:05d} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f} | "
              f"Test acc: {test_acc:.4f} | Patience: {earlystop.counter}/{patience}")
        if earlystop.early_stop:
            print("Early Stopping!")
            break
    earlystop.load_model(model)
    acc = evaluate(g, feat, labels, test_mask, model)
    print("Test accuracy {:.4f}".format(acc))
    test_acc_list.append(acc)
    return train_acc_list, valid_acc_list, test_acc_list, acc