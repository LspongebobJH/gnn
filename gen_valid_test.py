from argparse import ArgumentParser
import pickle
import os

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--version", type=int, default=1)

    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'dataset', 'valid_test_split')
    path = os.path.join(dir_path, f'v{args.version}.pkl')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_path = './dataset'
        path = os.path.join(data_path, 'FC_Fisher_Z_transformed.pkl')
        with open(path, 'rb') as f:
            data_FC = pickle.load(f)

        path = os.path.join(data_path, 'SC.pkl')
        with open(path, 'rb') as f:
            data_SC = pickle.load(f)

        path = os.path.join(data_path, 'T1.pkl')
        with open(path, 'rb') as f:
            data_raw_X = pickle.load(f)

        path = os.path.join(data_path, 'demo.pkl')
        with open(path, 'rb') as f:
            data_labels = pickle.load(f)
        
        # NOTE note that only data_SC has full keys, is it a semi-supervised task?
        ## we only take graph which have labels and both SC, FC modalities.
        adjs = torch.zeros((len(data_labels), 2, 200, 200))
        labels = torch.zeros(len(data_labels))
        raw_Xs = torch.zeros((len(data_labels), 200, 9))
        mask = []
        for i, name in tqdm(enumerate(data_labels.keys())):
            if name not in data_SC.keys() or name not in data_FC.keys() or name not in data_raw_X.keys():
                continue
            if 'nih_totalcogcomp_ageadjusted' not in data_labels[name].keys():
                continue
            adjs[i, 0] = torch.tensor(data_SC[name])
            adjs[i, 1] = torch.tensor(data_FC[name])

            labels[i] = float(data_labels[name]['nih_totalcogcomp_ageadjusted'])

            raw_X = data_raw_X[name].drop(columns=['StructName'])
            raw_X = raw_X.to_numpy().astype(float)
            raw_Xs[i] = torch.tensor(raw_X)

            mask.append(i)
        adjs = adjs[mask]
        labels = labels[mask]
        raw_Xs = raw_Xs[mask]

        mu_lbls, std_lbls = None, None
        if label_type == 'classification':
            labels_class = torch.zeros_like(labels, dtype=torch.long)
            for i, label in enumerate(labels.unique()):
                labels_class[labels == label] = i
            labels = labels_class

        else:
            mu_lbls, std_lbls = labels.mean(), labels.std()
            labels = (labels - mu_lbls) / std_lbls
        
        batchnorm = nn.BatchNorm1d(raw_Xs.shape[-1], affine=False)
        layernorm = nn.LayerNorm([adjs.shape[-2], adjs.shape[-1]], elementwise_affine=False)

        original_feat_shape = raw_Xs.shape
        raw_Xs = batchnorm(
            raw_Xs.reshape(-1, raw_Xs.shape[-1])
        ).reshape(original_feat_shape)

        original_adjs_shape = adjs.shape
        adjs = layernorm(
            adjs.reshape(-1, adjs.shape[-2], adjs.shape[-1])
        ).reshape(original_adjs_shape)

        data = {
            'adjs': adjs,
            'raw_Xs': raw_Xs,
            'labels': labels,
            'mu_lbls': mu_lbls,
            'std_lbls': std_lbls,
        }

        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

        assert (adjs == torch.transpose(adjs, 2, 3)).all().item(), "adj matrices are not symmetric"

    if eval_type == 'split':
        train_size, valid_size, test_size = \
            split_args['train_size'], split_args['valid_size'], split_args['test_size']
        idx = np.arange(len(labels))
        train_valid_idx, test_idx = \
            train_test_split(idx, test_size=test_size)
        train_idx, valid_idx = \
            train_test_split(train_valid_idx, 
                            test_size=valid_size / (train_size + valid_size))
        splits = {
            'train_idx': train_idx,
            'valid_idx': valid_idx,
            'test_idx': test_idx,
        }
    elif eval_type == 'cross':
        kfold = KFold(n_splits=5, shuffle=True)
        splits = list(kfold.split(X=idx))