import torch
import torch.utils.data

from tqdm import tqdm


class Trainer:
    def __init__(self, model, loss_func, optimizer, metric, train_loader, valid_loader, test_loader, config):
        self.config = config
        print('=' * 10 + "Config" + '=' * 10)
        for k, v in self.config.items():
            print(f'{k}: {v}')
        print('=' * 25)
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.metric = metric
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.to(self.config['DEVICE'])

    def to(self, device=None):
        if device is None:
            self.model = self.model.to(self.config['DEVICE'])
            self.loss_func = self.loss_func.to(self.config['DEVICE'])
        else:
            self.model = self.model.to(device)
            self.loss_func = self.loss_func.to(self.config['DEVICE'])
            self.config['DEVICE'] = device

    def step(self, batch_data, mode='train', **param_for_rec):
        device = self.config['DEVICE']

        def compute_pred():
            if type(batch_data) == tuple:
                batch_y = batch_data[0].unsqueeze(1).to(device)
                batch_x = batch_data[1].to(device)

                pred = self.model(batch_x)
                loss = self.loss_func(pred, batch_y)
            elif type(batch_data) == list:
                pred = self.model(batch_data)
                loss = self.loss_func(pred, batch_data[1].coalesce().values())

            return loss, pred

        if mode == 'train':
            self.model.train()
            self.optimizer.zero_grad()
            loss, pred = compute_pred()
            loss.backward()
            self.optimizer.step()
            return loss.item(), pred
        elif mode == 'evaluate':
            with torch.no_grad():
                self.model.eval()
                loss, pred = compute_pred()
                if self.config['TASK'] == 'classification':
                    self.metric.compute_metric(pred, batch_data[0])
                elif self.config['TASK'] == 'recommend':
                    gt_rec_lst = param_for_rec['gt_rec_lst']
                    pred_rec_lst = param_for_rec['pred_rec_lst']

                    user_idx_lst = batch_data[0]
                    sparse_indices = batch_data[1].coalesce().indices()
                    for i in range(len(user_idx_lst)):
                        tmp_idx = torch.nonzero(sparse_indices[0, :] == i).view(-1)
                        gt_rec_lst.append(sparse_indices[1, tmp_idx].tolist())
                    pred_rec_lst.extend(self.model.rec(user_idx_lst)[1].tolist())
                return loss.item(), pred, gt_rec_lst, pred_rec_lst
        else:
            raise ValueError("Wrong Mode")

    def _compute_metric(self, metric_str, **param_for_rec):
        if self.config['TASK'] == 'classification':
            self.metric.get_batch_metric()
            for k, v in self.metric.metric_dict.items():
                if k == 'cnt':
                    continue
                metric_str += f'{k}: {v:4f}\n'
            self.metric.init_metric()
            return metric_str
        elif self.config['TASK'] == 'recommend':
            gt_rec_lst = param_for_rec['gt_rec_lst']
            pred_rec_lst = param_for_rec['pred_rec_lst']
            self.metric.compute_metric(gt_rec_lst, pred_rec_lst, self.config['ITEM_NUM'], self.config['USER_NUM'])

            metric_string = ""
            for m in self.metric.metric_dict.keys():
                metric_string += m + ':\n'
                for small_m in self.metric.metric_dict[m].keys():
                    metric_string += '\t' + small_m + f': {self.metric.metric_dict[m][small_m]:.4f}' + '\n'
            return metric_string

    def train(self):
        print("=" * 10 + "TRAIN BEGIN" + "=" * 10)
        epoch = self.config['EPOCH']
        for e in range(1, epoch + 1):
            all_loss = 0.0
            for s, batch_data in enumerate(tqdm(self.train_loader)):
                loss, pred = self.step(batch_data, mode='train')
                all_loss += loss

            all_loss /= s + 1
            print(f'Train Epoch: {e}\nLoss: {all_loss}')
            if e % 1 == 0:
                all_loss = 0.0
                self.metric.init_metric()
                gt_rec_lst = []
                pred_rec_lst = []
                for s, batch_data in enumerate(tqdm(self.valid_loader)):
                    res = self.step(batch_data, mode='evaluate', gt_rec_lst=gt_rec_lst, pred_rec_lst=pred_rec_lst)
                    if len(res) == 2:
                        loss, pred = res
                    elif len(res) == 4:
                        loss, pred, gt_rec_lst, pred_rec_lst = res
                    all_loss += loss

                all_loss /= s + 1
                metric_str = f'loss: {all_loss}\n'

                metric_str = self._compute_metric(metric_str, gt_rec_lst=gt_rec_lst, pred_rec_lst=pred_rec_lst)

                print(f'Valid Epoch: {e}\n' + metric_str)
        print("=" * 10 + "TRAIN END" + "=" * 10)

    def test(self):
        all_loss = 0.0
        self.metric.init_metric()
        gt_rec_lst = []
        pred_rec_lst = []
        for s, batch_data in enumerate(tqdm(self.test_loader)):
            res = self.step(batch_data, mode='evaluate', gt_rec_lst=gt_rec_lst, pred_rec_lst=pred_rec_lst)
            if len(res) == 2:
                loss, pred = res
            elif len(res) == 4:
                loss, pred, gt_rec_lst, pred_rec_lst = res
            all_loss += loss

        all_loss /= s + 1
        metric_str = f'loss: {all_loss}\n'
        metric_str = self._compute_metric(metric_str, gt_rec_lst=gt_rec_lst, pred_rec_lst=pred_rec_lst)
        print(f'Test Loss: {all_loss}\n' + metric_str)
