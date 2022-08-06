import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.utils.data

from collections import Counter

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