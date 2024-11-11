from DMGI_src import AvgReadout, modeler, process, evaluate
from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np
import time
from utils import load_dataset, adj_weight2bin
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from utils import set_random_seed, load_dataset, model_infer, Evaluator, EarlyStopping

class embedder:
    def __init__(self, args):
        args.sparse = True
        args.metapaths_list = args.metapaths.split(",")

        # adj, features, labels, idx_train, idx_valid, idx_test = process.load_data_dblp(args)
        adj, features, labels, splits = load_dataset(split_args=args.split_args, label_type=args.label_type)
        idx_train, idx_valid, idx_test = \
            splits['train_idx'], splits['valid_idx'], splits['test_idx']

        args.nb_graphs = len(adj)
        args.nb_nodes = features.shape[1]
        args.ft_size = features.shape[-1]
        args.nb_classes = 1
        args.nb_modal = adj.shape[1]

        if args.preprocess == 'author':
            features = [process.preprocess_features(feature) for feature in features]
            self.features = [torch.FloatTensor(feature[np.newaxis]) for feature in features]
            adj = [process.normalize_adj(adj_) for adj_ in adj]
            self.adj = [process.sparse_mx_to_torch_sparse_tensor(adj_) for adj_ in adj]

        elif args.preprocess == 'our':
            batch_norm = nn.BatchNorm1d(features.shape[-1], affine=False)
            layer_norm = nn.LayerNorm(normalized_shape=(adj.shape[-2], adj.shape[-1]), elementwise_affine=False)
            original_shape = features.shape
            self.features = batch_norm(features.reshape(-1, features.shape[-1])).reshape(original_shape)
            adj = layer_norm(adj)
            adj_0, adj_1 = adj_weight2bin(adj, args.ratio, args.ratio)

        data_list_0, data_list_1 = [], []
        for i in tqdm(range(len(adj_0))):
            data_0 = {
                'edge_index': torch.stack(torch.nonzero(adj_0[i], as_tuple=True)),
                'num_nodes': features.shape[1]
            }
            data_1 = {
                'edge_index': torch.stack(torch.nonzero(adj_1[i], as_tuple=True)),
                'num_nodes': features.shape[1]
            }
            data_0 = Data(**data_0)
            data_1 = Data(**data_1)
            data_list_0.append(data_0)
            data_list_1.append(data_1)

        data_train_list, data_valid_list, data_test_list = [], [], []
        for data_list in [data_list_0, data_list_1]:
            data_train = Batch.from_data_list(
                            [data_list[i] for i in idx_train]
                        ).to(args.device)
            data_valid = Batch.from_data_list(
                            [data_list[i] for i in idx_valid]
                        ).to(args.device)
            data_test = Batch.from_data_list(
                            [data_list[i] for i in idx_test]
                        ).to(args.device)
            data_train_list.append(data_train)
            data_valid_list.append(data_valid)
            data_test_list.append(data_test)

        self.data_train = data_train_list
        self.data_valid = data_valid_list
        self.data_test = data_test_list

        self.labels = labels.to(args.device)
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test

        self.train_lbls = self.labels[self.idx_train]
        self.val_lbls = self.labels[self.idx_valid]
        self.test_lbls = self.labels[self.idx_test]

        args.nb_train, args.nb_valid, args.nb_test = \
            len(self.train_lbls), len(self.val_lbls), len(self.test_lbls)

        # How to aggregate
        args.readout_func = AvgReadout()

        # Summary aggregation
        args.readout_act_func = nn.Sigmoid()

        self.args = args

    def currentTime(self):
        now = time.localtime()
        s = "%04d-%02d-%02d %02d:%02d:%02d" % (
            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

        return s
    
class DMGI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        evaluator = Evaluator(label_type='regression', num_classes=1, device=self.args.device)
        features = self.features.to(args.device)
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        # xent = nn.CrossEntropyLoss()
        xent = nn.MSELoss()
        for epoch in range(self.args.nb_epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()

            # shuffle feature dimension
            # idx = np.stack(
            #     [np.random.permutation(self.args.ft_size) for _ in 
            #      range(features.shape[0] * features.shape[1])]
            # )
            # idx = torch.tensor(idx, device=features.device)
            # original_shape = features.shape
            # shuf = torch.gather(
            #     features.reshape(-1, original_shape[-1]), 1, idx
            # ).reshape(original_shape)
            
            # shuffle node dimension
            shuf = torch.stack(
                [features[i][torch.randperm(features.shape[1])] for i in range(len(features))]
            )
            # shuf = [feature[:, idx, :] for feature in features]
            # shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(len(self.data_train[0]), self.args.nb_nodes)
            lbl_2 = torch.zeros(len(self.data_train[0]), self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)
            
            result = model(features[self.idx_train], self.data_train, shuf[self.idx_train], self.args.sparse, None, None, None)
            logits = result['logits']

            for view_idx, logit in enumerate(logits):
                if xent_loss is None:
                    xent_loss = b_xent(logit, lbl)
                else:
                    xent_loss += b_xent(logit, lbl)

            loss = xent_loss

            reg_loss = result['reg_loss']
            loss += self.args.reg_coef * reg_loss

            if self.args.isSemi:
                sup = result['semi']
                semi_loss = xent(sup, self.train_lbls)
                loss += self.args.sup_coef * semi_loss

            print(f"Epoch {epoch} | Train MSE Loss {loss.item(): .4f}")

            loss.backward()
            optimiser.step()

            if epoch % 5 == 0:
                model.eval()
                logits_valid = model(features[self.idx_valid], self.data_valid, shuf[self.idx_valid], self.args.sparse, None, None, None)['semi']
                logits_test = model(features[self.idx_test], self.data_test, shuf[self.idx_test], self.args.sparse, None, None, None)['semi']
                valid_rmse = evaluator.evaluate(logits_valid, self.val_lbls)
                test_rmse = evaluator.evaluate(logits_test, self.test_lbls)
                print(f"Valid RMSE {valid_rmse:.4f} | Test RMSE {test_rmse:.4f}")

            # if loss < best:
            #     best = loss
            #     cnt_wait = 0
            #     torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
            # else:
            #     cnt_wait += 1

            # if cnt_wait == self.args.patience:
            #     break

            

        pass
        # model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths)))

        # Evaluation
        # model.eval()
        # evaluate(model.H.data.detach(), self.idx_train, self.idx_valid, self.idx_test, self.labels, self.args.device)

if __name__ == '__main__':
    data = {
        "embedder": "DMGI",
        "dataset": "imdb",
        "metapaths": "MAM,MDM",
        "nb_epochs": 1000,
        "hid_units": 64,
        "lr": 1e-3,
        "l2_coef": 0.0001,
        "drop_prob": 0.5,
        "reg_coef": 0.001,
        "sup_coef": 0.1,
        "sc": 3.0,
        "margin": 0.1,
        "patience": 20,
        "nheads": 1,
        "activation": "relu",
        "isSemi": True,
        "isBias": False,
        "isAttn": False,
        "preprocess": "our", # [our, author]
        "label_type": "regression",
        "split_args": {
                    'train_size': 0.6,
                    'valid_size': 0.2,
                    'test_size': 0.2,
                },
        "ratio": 0.3,
        "device": "cuda:7"
    }
    args = Namespace(**data)

    model = DMGI(args)
    model.training()
