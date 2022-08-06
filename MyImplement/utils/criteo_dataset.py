import numpy as np
import pandas as pd
import pandas_profiling as pp
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder

import torch
import torch.utils.data
from torch.utils.data import Dataset

import os


class CriteoDataset(Dataset):
    def __init__(self, pth, mode='train', encoders=None, show_info=False):
        self.pth = pth
        self.mode = mode
        self.encoders = encoders
        super(CriteoDataset, self).__init__()
        dtype = {
            'label': np.int32,
        }
        for i in range(13):
            dtype[f'I_{i + 1}'] = np.float32
        for i in range(26):
            dtype[f'C_{i + 1}'] = str

        with open(pth, 'r', encoding='utf-8') as file:
            data_df = pd.read_csv(file, sep='\t', dtype=dtype)
            data_df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

        self.data_df: pd.DataFrame = data_df
        self.encoders = self._pre_process()
        self.data_X = torch.tensor(self.data_df.values[:, 1:], dtype=torch.float32)
        self.data_y = torch.tensor(self.data_df.values[:, 0], dtype=torch.float32)
        if show_info:
            print(self.data_df.info())
            print(self.data_df.head())

    def _generate_report(self):
        report_pth = os.path.split(self.pth)[0] + '/' + f'Criteo-{self.mode}-report.html'
        if not os.path.exists(report_pth):
            report = pp.ProfileReport(self.data_df)
            report.to_file(report_pth)

    def __getitem__(self, index):
        return self.data_y[index], self.data_X[index]

    def __len__(self):
        return self.data_X.shape[0]

    def _pre_process(self):
        # 要注意一点，这里想要使用的目标编码方式是有监督的，或者是利用了全局的统计信息，这就牵扯到了数据集划分的问题
        # 应该只利用训练集上的监督信息对其进行编码，所以需要在预处理前先划分好三个数据集
        one_hot_columns = [f'C_{i}' for i in [6, 9, 14, 17, 20, 22, 23]]
        target_columns = [f'C_{i}' for i in range(1, 27) if i not in [6, 9, 14, 17, 20, 22, 23]]
        if self.encoders is None:
            # 数值型特征使用均值填充
            mean_fillna = dict()
            for i in range(1, 14):
                m = self.data_df[f'I_{i}'].mean()
                mean_fillna[f'I_{i}'] = m
                self.data_df[f'I_{i}'] = self.data_df[f'I_{i}'].fillna(m)

            # 根据分析报告，c6 c9 c14 c17 c20 c22 c23 这些特征类型数量在20以内，使用onehot编码
            one_hot_encoder = OneHotEncoder(cols=one_hot_columns, handle_unknown='value', handle_missing='value').fit(
                self.data_df)
            self.data_df = one_hot_encoder.transform(self.data_df)

            target_encoder = TargetEncoder(cols=target_columns, handle_unknown='value', handle_missing='value').fit(
                self.data_df.drop(columns=['label']), self.data_df['label'])
            label = self.data_df['label']
            self.data_df = target_encoder.transform(self.data_df.drop(columns=['label']))
            self.data_df = pd.concat([label, self.data_df], axis=1)

            return [mean_fillna, one_hot_encoder, target_encoder]
        else:
            mean_fillna = self.encoders[0]
            for i in range(1, 14):
                m = mean_fillna[f'I_{i}']
                self.data_df[f'I_{i}'] = self.data_df[f'I_{i}'].fillna(m)

            one_hot_encoder = self.encoders[1]
            target_encoder = self.encoders[2]
            self.data_df = one_hot_encoder.transform(self.data_df)
            label = self.data_df['label']
            self.data_df = target_encoder.transform(self.data_df.drop(columns=['label']))
            self.data_df = pd.concat([label, self.data_df], axis=1)