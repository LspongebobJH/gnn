from DMGI_src import modeler, embedder, evaluate
from argparse import Namespace
import torch
import torch.nn as nn
import numpy as np

class DMGI(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args

    def training(self):
        features = [feature.to(self.args.device) for feature in self.features]
        adj = [adj_.to(self.args.device) for adj_ in self.adj]
        model = modeler(self.args).to(self.args.device)
        optimiser = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.l2_coef)
        cnt_wait = 0; best = 1e9
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        for epoch in range(self.args.nb_epochs):
            xent_loss = None
            model.train()
            optimiser.zero_grad()
            idx = np.random.permutation(self.args.nb_nodes)

            shuf = [feature[:, idx, :] for feature in features]
            shuf = [shuf_ft.to(self.args.device) for shuf_ft in shuf]

            lbl_1 = torch.ones(self.args.batch_size, self.args.nb_nodes)
            lbl_2 = torch.zeros(self.args.batch_size, self.args.nb_nodes)
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.args.device)

            result = model(features, adj, shuf, self.args.sparse, None, None, None)
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
                semi_loss = xent(sup[self.idx_train], self.train_lbls)
                loss += self.args.sup_coef * semi_loss

            if loss < best:
                best = loss
                cnt_wait = 0
                torch.save(model.state_dict(), 'saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths))
            else:
                cnt_wait += 1

            if cnt_wait == self.args.patience:
                break

            loss.backward()
            optimiser.step()


        model.load_state_dict(torch.load('saved_model/best_{}_{}_{}.pkl'.format(self.args.dataset, self.args.embedder, self.args.metapaths)))

        # Evaluation
        model.eval()
        evaluate(model.H.data.detach(), self.idx_train, self.idx_val, self.idx_test, self.labels, self.args.device)

if __name__ == '__main__':
    data = {
        "embedder": "DMGI",
        "dataset": "imdb",
        "metapaths": "MAM,MDM",
        "nb_epochs": 10000,
        "hid_units": 64,
        "lr": 0.0005,
        "l2_coef": 0.0001,
        "drop_prob": 0.5,
        "reg_coef": 0.001,
        "sup_coef": 0.1,
        "sc": 3.0,
        "margin": 0.1,
        "gpu_num": 0,
        "patience": 20,
        "nheads": 1,
        "activation": "relu",
        "isSemi": False,
        "isBias": False,
        "isAttn": False
    }
    args = Namespace(**data)

    model = DMGI(args)
    model.training()
