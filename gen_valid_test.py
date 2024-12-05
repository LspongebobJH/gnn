from argparse import ArgumentParser
import pickle
import os
from tqdm import tqdm
import numpy as np
from utils import set_random_seed

from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--version", type=int, default=1)

    args = parser.parse_args()

    split_args = {
        'train_size': 0.6,
        'valid_size': 0.2,
        'test_size': 0.2
    }

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
    
    names = []
    for i, name in tqdm(enumerate(data_labels.keys())):
        if name not in data_SC.keys() or name not in data_FC.keys() or name not in data_raw_X.keys():
            continue
        if 'nih_totalcogcomp_ageadjusted' not in data_labels[name].keys():
            continue

        names.append(name)

    names = np.array(names)
    train_size, valid_size, test_size = \
        split_args['train_size'], split_args['valid_size'], split_args['test_size']
    idx = np.arange(len(names))
    splits_seeds = []
    for seed in range(10):
        set_random_seed(seed)
        train_valid_idx, test_idx = \
            train_test_split(idx, test_size=test_size)
        train_idx, valid_idx = \
            train_test_split(train_valid_idx, 
                            test_size=valid_size / (train_size + valid_size))
        valid_names, test_names = names[valid_idx], names[test_idx]
        splits = {
            'valid_names': valid_names,
            'test_names': test_names,
        }
        splits_seeds.append(splits)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.join(dir_path, 'dataset', 'valid_test_split')
    path = os.path.join(dir_path, f'v{args.version}.pkl')

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(path, 'wb') as f:
        pickle.dump(splits_seeds, f)
