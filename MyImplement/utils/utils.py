import numpy as np
import pandas as pd
import pandas_profiling as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from category_encoders.target_encoder import TargetEncoder
from category_encoders.one_hot import OneHotEncoder

import torch
import torch.utils.data
from torch.utils.data import Dataset

from collections import Counter
import os


class MyDataset:
    def __init__(self):
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


class Metric:
    def __init__(self, k=(1, 2, 3)):
        self.k = k
        self._init_metric()

    def _init_metric(self):
        hit = dict()
        gini_index = dict()
        diversity = dict()
        nDCG = dict()
        MRR = dict()
        for _k in self.k:
            hit[f'hit@{_k}'] = 0.0
            gini_index[f'gini_index@{_k}'] = 0.0
            diversity[f'diversity@{_k}'] = 0.0
            nDCG[f'nDCG@{_k}'] = 0.0
            MRR[f'MRR@{_k}'] = 0.0

        self.metric_dict = {
            'hit': hit,
            'gini_index': gini_index,
            'diversity': diversity,
            'nDCG': nDCG,
            'MRR': MRR
        }

    def _hit(self, label, pred):
        assert len(label) == len(pred)
        for _k in self.k:
            length = len(label)
            cnt = 0
            for j in range(len(label)):
                if len(label[j]) == 0:
                    length -= 1
                    continue
                if len(set(label[j]) & set(pred[j][:_k])) > 0:
                    cnt += 1
            self.metric_dict['hit'][f'hit@{_k}'] = cnt / length

    def _gini_index(self, label, pred):
        length = len(pred)
        for _k in self.k:
            # 计算流行度，需要把对label中没有出现的用户的预测忽略掉
            item_cnt = dict()
            total_cnts = 0
            for j in range(len(pred)):
                if len(label[j]) == 0:
                    continue
                new_dict = Counter(pred[j][:_k])
                temp = {x: item_cnt[x] + new_dict[x] for x in item_cnt if x in new_dict}
                item_cnt.update(Counter(pred[j][:_k]))
                item_cnt.update(temp)
                total_cnts += len(pred[j][:_k])

            actual_area = 0
            last_w_i = 0
            for item, popularity in sorted(item_cnt.items(), key=lambda d: d[1]):
                popularity = last_w_i + popularity
                actual_area += (last_w_i / total_cnts + popularity / total_cnts) / (2 * len(item_cnt.keys()))
                last_w_i = popularity
            gini_index = (1 / 2 - actual_area) * 2
            self.metric_dict['gini_index'][f'gini_index@{_k}'] = gini_index

    def _diversity(self, label, pred):
        # hamming distance
        for _k in self.k:
            n = len(pred)
            H_ij = 0.0
            for i in range(len(pred)):
                if len(label[i]) == 0:
                    n -= 1
                    continue
                for j in range(i + 1, len(pred)):
                    if len(label[j]) == 0:
                        continue
                    Q_ij = len(set(pred[i][:_k]) & set(pred[j][:_k]))
                    H_ij += 1 - Q_ij / _k
            self.metric_dict['diversity'][f'diversity@{_k}'] = H_ij / (n * (n - 1) / 2)

    def _create_implict_matrix(self, label, n_items, n_users):
        assert len(label) == n_users
        rel_matrix = [[0] * n_items for _ in range(n_users)]
        for i in range(n_users):
            for j in label[i]:
                # print(f'i:{i},j:{j}')
                rel_matrix[i][j - 1] = 1
        self.rel_matrix = np.array(rel_matrix)

    def _nDCG(self, label, pred):
        if type(pred) is not np.ndarray:
            raise TypeError("pred不是numpy数组")
        for _k in self.k:
            n = len(pred)
            nDCG = 0
            for i in range(len(pred)):
                if len(label[i]) == 0:
                    n -= 1
                    continue
                # print(pred[i])
                # print(type(pred), pred.shape)
                # print(type(pred[i]))
                DCG = np.sum(self.rel_matrix[i, pred[i][:_k] - 1] / np.log(np.array(range(1, _k + 1)) + 1))
                iDCG = np.sum(
                    np.sort(self.rel_matrix[i, pred[i][:_k] - 1])[::-1] / np.log(np.array(range(1, _k + 1)) + 1))
                if iDCG != 0:
                    nDCG += DCG / iDCG
            nDCG /= n
            self.metric_dict['nDCG'][f'nDCG@{_k}'] = nDCG

    def _MRR(self, label, pred):
        for _k in self.k:
            MRR = 0.0
            n = len(label)
            for i in range(len(label)):
                if len(label[i]) == 0:
                    n -= 1
                    continue
                for j in range(_k):
                    if pred[i][j] in label[i]:
                        MRR += 1 / (j + 1)
                        break
            MRR /= n
            self.metric_dict['MRR'][f'MRR@{_k}'] = MRR

    def __str__(self):
        metric_string = ""
        for m in self.metric_dict.keys():
            metric_string += m + ':\n'
            for small_m in self.metric_dict[m].keys():
                metric_string += '\t' + small_m + f': {self.metric_dict[m][small_m]:.4f}' + '\n'
        return metric_string

    def compute_metric(self, label, pred, n_items, n_users):
        # 目前在整体pred上计算
        assert len(label) == len(pred)
        self._create_implict_matrix(label, n_items, n_users)
        self._hit(label, pred)
        self._gini_index(label, pred)
        self._diversity(label, pred)
        self._nDCG(label, pred)
        self._MRR(label, pred)


class CTRMetric:
    def __init__(self):
        self.metric_dict = dict()
        self.init_metric()

    def init_metric(self):
        self.metric_dict = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'F1': 0.0,
            'AUC': 0.0,
            'cnt': 0,
        }

    def compute_metric(self, pred: torch.Tensor, label: torch.Tensor, threshold=0.5):
        pred = pred.detach().cpu().numpy().flatten()
        label = label.detach().cpu().numpy().flatten()

        self.metric_dict['AUC'] += roc_auc_score(label, pred)

        pred = pred > threshold
        label = label == 1

        TP = np.sum(pred * label)
        TN = np.sum((~pred) * (~label))
        FP = np.sum(pred * (~label))
        FN = np.sum((~pred) * label)

        T = np.sum(label)
        F = label.shape[0] - T

        self.metric_dict['accuracy'] += (TP + TN) / (T + F)
        P = TP / (TP + FP) if TP + FP != 0 else 0
        self.metric_dict['precision'] += P
        R = TP / (TP + FN) if TP + FN != 0 else 0
        self.metric_dict['recall'] += R
        self.metric_dict['F1'] += 2 / (1 / P + 1 / R) if P * R != 0 else 0

        self.metric_dict['cnt'] += 1

    def get_batch_metric(self):
        cnt = self.metric_dict['cnt']
        for k, v in self.metric_dict.items():
            if k == 'cnt':
                continue
            self.metric_dict[k] = v / cnt

class Logger:
    def __init__(self):
        pass

    def log(self):
        pass

    def __str__(self):
        pass


def dataset_test():
    # MovieLens
    # with open('../../dataset/ml-latest-small-ratings.txt', 'r', encoding='utf-8') as f:
    #     ml_dataset = MovieLensDataset(f)
    #     print(ml_dataset.data[:10, :])
    #     print(type(ml_dataset.data))
    # field_dims, train_set, valid_set, test_set = ml_dataset.train_valid_test_split(train_size=0.8, valid_size=0.1,
    #                                                                                test_size=0.1)
    # print(f'train_set shape: {train_set[0].shape}')
    # print(f'valid_set shape: {valid_set[0].shape}')
    # print(f'test_set shape: {test_set[0].shape}')

    # Criteo
    train_pth = '../../dataset/criteo-100k-train.txt'
    test_pth = '../../dataset/criteo-100k-test.txt'
    train_set = CriteoDataset(train_pth, mode='train')

    encoders = train_set.encoders

    test_set = CriteoDataset(test_pth, mode='test', encoders=encoders)

    print(test_set[123])


def metric_test():
    metric = Metric()
    label = [
        [3, 2, 1],
        [],
        [1, 3, 2]
    ]
    pred = np.array([
        [3, 2, 1],
        [2, 1, 3],
        [1, 3, 2]
    ])
    # pred = np.random.randint(1, 11, (5, 50))
    print(pred)
    metric.compute_metric(label, pred, 3, 3)
    print(metric.metric_dict)


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


def ctr_metric_test():
    pred = torch.tensor(
        [[0.8, 0.7, 0.5, 0.5, 0.5, 0.5, 0.3]],
        dtype=torch.float32,
        device='mps'
    )
    label = torch.tensor(
        [[1, 1, 0, 0, 1, 1, 0]],
        dtype=torch.float32,
        device='mps'
    )

    ctr_metric = CTRMetric()
    ctr_metric.compute_metric(pred, label)
    print(ctr_metric.metric_dict)

    ctr_metric.compute_metric(pred, label)
    ctr_metric.get_batch_metric()
    print(ctr_metric.metric_dict)



if __name__ == '__main__':
    # read_and_split_data('../../dataset/criteo-100k.txt')
    ctr_metric_test()
