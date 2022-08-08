import numpy as np
import pandas as pd
import torch.nn as nn
import os


class Logger:
    def __init__(self):
        pass

    def log(self):
        pass

    def __str__(self):
        pass


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.zeros_(m.bias.data)
    elif isinstance(m, nn.Embedding):
        nn.init.kaiming_normal_(m.weight.data)


def read_and_split_data(pth, read_part=False, sample_number=100000, size=None, random_state=2022):
    if size is None:
        size = [0.8, 0.1, 0.1]

    names = [
        'label',
    ]
    dtype = {
        'label': np.int32,
    }
    for i in range(13):
        names.append(f'I_{i + 1}')
        dtype[f'I_{i + 1}'] = np.float32
    for i in range(26):
        names.append(f'C_{i + 1}')
        dtype[f'C_{i + 1}'] = str

    train_size = size[0]
    valid_size = size[1]
    test_size = size[2]

    with open(pth, 'r', encoding='utf-8') as file:
        if read_part:
            data_df = pd.read_csv(file, sep='\t', names=names, dtype=dtype, nrows=sample_number)
        else:
            data_df = pd.read_csv(file, sep='\t', names=names, dtype=dtype)

    train_df = data_df.sample(frac=train_size, random_state=random_state, axis=0)
    valid_test = data_df[~data_df.index.isin(train_df.index)]

    valid_size = valid_size / (valid_size + test_size)
    valid_df = valid_test.sample(frac=valid_size, random_state=random_state, axis=0)
    test_df = valid_test[~valid_test.index.isin(valid_df.index)]

    print(test_df.info())
    print(test_df.head())
    save_pth = os.path.split(pth)[0]
    train_save_pth = save_pth + '/criteo-100k-train.txt'
    valid_save_pth = save_pth + '/criteo-100k-valid.txt'
    test_save_pth = save_pth + '/criteo-100k-test.txt'

    train_df.to_csv(train_save_pth, sep='\t')
    valid_df.to_csv(valid_save_pth, sep='\t')
    test_df.to_csv(test_save_pth, sep='\t')
