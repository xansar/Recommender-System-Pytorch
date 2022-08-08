import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data
from torch.utils.data import Dataset


class MovieLensDatasetWithTorch(Dataset):
    def __init__(self, file, task='rating', read_part=True, sample_number=100000):
        super(MovieLensDatasetWithTorch, self).__init__()

        dtype = {
            'userID': np.int32,
            'MovieID': np.int32,
            'rating': np.float16
        }

        if read_part:
            data_df = pd.read_csv(file, sep=',', dtype=dtype, nrows=sample_number)
        else:
            data_df = pd.read_csv(file, sep=',', dtype=dtype)

        data_df = data_df.drop(columns=['timestamp'])

        if task == 'classification':
            data_df['rating'] = data_df.apply(lambda x: 1 if x['rating'] > 3.0 else 0, axis=1)

        tmp = torch.tensor(data_df.values)
        i = tmp[:, :2].t()
        [user_num, item_num], _ = torch.max(i, dim=1)
        self.user_num = user_num.long() + 1
        self.item_num = item_num.long() + 1
        v = tmp[:, 2]

        self.user_item_matrix = torch.sparse_coo_tensor(i, v, (self.user_num, self.item_num), dtype=torch.float32)

    def __getitem__(self, index):
        return [index, self.user_item_matrix[index]]

    def __len__(self):
        return self.user_item_matrix.shape[0]


class MyDataset(Dataset):
    def __init__(self):
        super(MyDataset, self).__init__()
        self.device = torch.device('cpu')
        self.data: pd.DataFrame = None

    def to(self, device):
        self.device = device
        return self

    def train_valid_test_split(self, train_size, valid_size, test_size, random_state=1):
        feild_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        train_set, valid_test = train_test_split(self.data, train_size=train_size, random_state=random_state)

        valid_size = valid_size / (valid_size + test_size)
        valid_set, test_set = train_test_split(valid_test, train_size=valid_size, random_state=random_state)

        device: torch.device = self.device

        train_X = torch.tensor(train_set[:, :-1], dtype=torch.long).to(device)
        valid_X = torch.tensor(valid_set[:, :-1], dtype=torch.long).to(device)
        test_X = torch.tensor(test_set[:, :-1], dtype=torch.long).to(device)
        train_Y = torch.tensor(train_set[:, -1], dtype=torch.long).to(device)
        valid_Y = torch.tensor(valid_set[:, -1], dtype=torch.long).to(device)
        test_Y = torch.tensor(test_set[:, -1], dtype=torch.long).to(device)

        return feild_dims, (train_X, train_Y), (valid_X, valid_Y), (test_X, test_Y)


class MovieLensDataset(MyDataset):
    def __init__(self, file, read_part=True, sample_number=100000, task='classification'):
        super(MovieLensDataset, self).__init__()

        dtype = {
            'userID': np.int32,
            'MovieID': np.int32,
            'rating': np.float16
        }

        if read_part:
            data_df = pd.read_csv(file, sep=',', dtype=dtype, nrows=sample_number)
        else:
            data_df = pd.read_csv(file, sep=',', dtype=dtype)

        data_df = data_df.drop(columns=['timestamp'])

        if task == 'classification':
            data_df['rating'] = data_df.apply(lambda x: 1 if x['rating'] > 3.0 else 0, axis=1)

        self.data = data_df.values
